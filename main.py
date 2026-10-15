def run(ktm_year:int):
    ## Establish upfront
    # 1. Stations lists
    # 2. Year index
    
    # ---- KTM Ridership Data ----
    
    # 1. Read in KTM Data
    ktm_df = get_clean_ktm_data(ktm_year)
    
    # 2. Establish the constants
    STATIONS_LIST = sorted(set(ktm_df['origin']) | set(ktm_df['destination']))
    DATETIME_INDEX = pd.date_range(f"{ktm_year}-01-01", f"{ktm_year}-12-31", freq="h")
    
    # 3. Convert tabular to numpy
    ktm_np = build_station_series(ktm_df, DATETIME_INDEX, STATIONS_LIST)
    
    # ---- Adjacency Matrix ----
    
    # 1. Establish OD flow
    od_flow = build_od_flow(ktm_df, STATIONS_LIST)
    
    adj_matrix = normalize_adjacency_matrix(build_adjacency_matrix(od_flow, 5_000))
    
    # ---- Standardisation and Train-test-split  ----
    x_all, y_all = make_windows(ktm_np, lookback)
    train_data, valid_data, test_data = split_train_valid_test(x_all, y_all)

    x_train, _ = train_data
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = x_mean, x_std

    train_scaled = scale_data(train_data, x_mean, x_std, y_mean, y_std)
    valid_scaled = scale_data(valid_data, x_mean, x_std, y_mean, y_std)
    test_scaled  = scale_data(test_data,  x_mean, x_std, y_mean, y_std)
    
    # ---- Fit Model ----
    
    # 1. Define model and optimiser
    model = STGNN(in_features=1, hidden_dim=64, num_gcn_layers=2, horizon=1, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. Fit
    history = fit(model=model,
        optimiser=opt,
        train_data = train_scaled,
        a_hat=a_hat,
        n_epoch=20,
        batch_size=50)
    
    result_dict = {
        "model":model,
        "history":history,
    }
    
    return result_dict