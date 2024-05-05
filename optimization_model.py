import torch
# 
# Create a weight tensor creates a weight tensor that has higher values for shorter horizons (i.e., earlier in the sequence).
#  It then applies these weights to outputs and batch_y before computing the loss.
#  This will result in the loss being more sensitive to errors in the shorter horizon predictions.
weights = torch.exp(torch.linspace(0, -1, steps=sequence_length)).to(device)

# Normalize the weights to sum to 1
weights = weights / weights.sum()

# Reshape to match the shape of outputs and batch_y
weights = weights.view(1, sequence_length, 1)

# Compute weighted loss
weighted_outputs = outputs * weights
weighted_batch_y = batch_y * weights
loss = criterion(weighted_outputs, weighted_batch_y)

# The rest of your code...
optimizer.zero_grad()
loss_total = loss_total + loss.item()
loss.backward()
optimizer.step()