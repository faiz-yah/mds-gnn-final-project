def build_station_series(
    ktm_df:pd.DataFrame,
    datetime_index:pd.DatetimeIndex,
    stations:list
) -> np.ndarray:
    
    """
    Convert KTM tabular data to numpy format.
    """
    
    agg_df = ktm_df.groupby([ktm_df['datetime'].dt.floor('h'), 'origin'])['ridership'].sum()
    
    # Pivot from long format to wide format
    # To fill 0 for stations without ridership at certain datetime
    # whereby other stations have data
    pivot_df = agg_df.unstack(fill_value=0) 
    
    # Reindex based on intialised value upstream (to ensure parity)
    # To fill 0 for stations that never appear as origin
    # or datetime (index) that has no data at all
    pivot_df = pivot_df.reindex(index=datetime_index, columns=stations, fill_value=0)
    
    # Convert to numpy
    numpy_df = pivot_df.to_numpy(dtype=np.float32)
    
    return numpy_df


### For Adjacency Matrix

def build_od_flow(ktm_df:pd.DataFrame,
                  STATIONS_LIST:list[str]
) -> np.ndarray :
    """
    Aggregate the daily tabular KTM data
    into a total annual ridership per origin-destination.
    
    Output format is {'origin: destination'}, ridership
    
    """
    
    # Obtain annual ridership for each origin-destination pair
    od_agg_df = ktm_df.groupby(['origin', 'destination'])['ridership'].sum()
    
    # Establish station index dict
    station_idx = {s:i for i, s in enumerate(STATIONS_LIST)}
    
    # Initialise empty numpy array to store the OD ridership
    od_flow = np.zeros((len(STATIONS_LIST), len(STATIONS_LIST)), dtype=np.float32)
    
    for (o,d), ridership in od_agg_df.items():
        od_flow[station_idx[o], station_idx[d]] = ridership
    
    return od_flow + od_flow.T

def build_adjacency_matrix(od_flow:np.ndarray, threshold:int)->np.ndarray:
    """
    Build Origin-Destination (OD) adjacency matrix.

    If the annual sum of an OD exceeds the defined threshold,
    we consider that the stations are connected.
    
    We represent this connection through a matrix of Origin station (row-level)
    to Destination station (column-level).
    
    If the connection is present, we fill an integer 1, else 0.
    """
    
    adj = (od_flow >= threshold).astype(np.float32)
    
    np.fill_diagonal(adj, 0) 

    return adj

def normalize_adjacency_matrix(
    adj_matrix: np.ndarray
)-> np.ndarray:
    """
    A_hat = D^-1/2 (A + I) D^-1/2  (Kipf & Welling GCN normalisation).
    """
    
    a_tilde = adj_matrix + np.eye(len(adj_matrix)) # 1. Add self-loops
    degree = a_tilde.sum(axis=1) # 2. Degree of each station
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree)) #3. D^-1/2 as a diagonal matrix
    
    return (d_inv_sqrt @ a_tilde @ d_inv_sqrt).astype(np.float32) # 4. Sandwich