import torch

checkpoint = torch.load('C:/Users/Utente/Desktop/Backup_3Dperception/final_results/multiple_step/30_epochs/weights/last.pt')                      
print("Epoch:", checkpoint['epoch'])