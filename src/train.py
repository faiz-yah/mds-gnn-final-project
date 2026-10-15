def fit(model: nn.Module, 
        optimiser: torch.optim.Optimizer, 
        train_data: tuple[np.ndarray, np.ndarray], 
        a_hat: np.ndarray,
        n_epoch:int,
        batch_size:int
)->dict[int,float]:
    
    # 1 Convert numpy to torch tensors (float32 is the expected format)
    x, y = train_data
    
    x = torch.tensor(x, dtype=torch.float32) # Convert to tensor [S,T,N,F]
    y = torch.tensor(y, dtype=torch.float32) # Convert to tensor [S,N,1]
    
    # Adjacency Matrix
    a_hat = torch.tensor(a_hat, dtype=torch.float32) # Convert to tensor [N,N]
    
    n_samples = len(x)
    loss_history = {}
    model.train() # turns dropout ON
    
    for epoch in range(n_epoch):
        order = torch.randperm(n_samples)
        epoch_loss = 0.0
        
        for start in range(0, n_samples, batch_size):
            idx = order[start: start+batch_size]
            x_batch, y_batch = x[idx], y[idx]
            
            # Forward pass
            y_pred = model(x_batch, a_hat)
                
            # Calculate loss
            loss = F.mse_loss(y_pred, y_batch)

            # Backprop (Calculate gradient)
            optimiser.zero_grad() # Clear the previous epoch's gradients, since PyTorch adds them up by default
            loss.backward() #dLoss/dWeight for every weight, stores in param.grad
        
            # Gradient Descent (Update weights)
            optimiser.step() # weight = weight - learning_rate * grad
            
            epoch_loss += loss.item() + len(idx) # weight by batch size (last batch may be smaller)

        
        loss_history[epoch] = epoch_loss / n_samples
        print(f"Epoch {epoch} of {n_epoch}: Loss is {loss_history[epoch]:.4f}")
    
    return loss_history