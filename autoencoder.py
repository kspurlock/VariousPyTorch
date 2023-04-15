import torch
from torch import nn
import torch.utils.data as D
import torch.nn.functional as F
import h5py

class Encoder(nn.Module):
    def __init__(self, input_shape, reduced_shape, device, dtype):
        super().__init__()

        self._linear_stack = nn.Sequential(
            nn.Linear(input_shape, reduced_shape),
            nn.ReLU(),
            nn.Linear(reduced_shape, reduced_shape),
            nn.ReLU(),
            ).to(device=device, dtype=dtype)
        
    def forward(self, x):
        return self._linear_stack(x)
    

class Decoder(nn.Module):
    def __init__(self, input_shape, reduced_shape, device, dtype):
        super().__init__()

        self._linear_stack = nn.Sequential(
            nn.Linear(reduced_shape, reduced_shape),
            nn.ReLU(),
            nn.Linear(reduced_shape, input_shape),
            nn.ReLU(),
            ).to(device=device, dtype=dtype)
        
    def forward(self, x):
        return self._linear_stack(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_shape, device="cuda:0", dtype=torch.double, reduction=0.50):
        super().__init__()
        self._reduced_dim = int(input_shape * reduction)

        self._latent_weights = nn.Linear(self._reduced_dim,
                                         self._reduced_dim)\
                                             .to(device=device, dtype=dtype)
                                             
        self.encoder = Encoder(input_shape, self._reduced_dim, device, dtype)
        self.decoder = Decoder(input_shape, self._reduced_dim, device, dtype)
    
    def forward(self, x):
        encode = self.encoder(x)
        latent = self._latent_weights(encode)
        decode = self.decoder(latent)
        return F.log_softmax(decode, dim=1)
    
    def fit(self,
            loader: D.DataLoader,
            optimizer,
            scheduler,
            criterion,
            epochs=100):
        
        history = []
        num_batches = loader.dataset.shape[0] // loader.batch_size
        
        for epoch in range(epochs):
            avg_loss = 0
            
            for bi, batch in enumerate(loader):
                # Forward
                loss = criterion(
                    self.forward(batch),
                    F.log_softmax(batch, dim=1)
                    )
                
                avg_loss += loss.item()
                
                # Make output
                num_bars = int(((bi / num_batches) * 20)) + 1
                completion_string = "="*num_bars
                completion_string += "-"*(20 - num_bars)

                print(
                    "Epoch: {} \t {} \t Train loss: {} Lr: {}"\
                    .format(epoch,
                            completion_string,
                            loss,
                            scheduler.get_last_lr()),
                    end="\r"
                    )
                
                # Backpropogate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Adjust LR
            scheduler.step()
            
            # Record
            avg_loss /= num_batches
            history.append(avg_loss)
            
        return history
    
    @torch.inference_mode()
    def encode(self, X):
        encode = self.encoder(X)
        latent = self._latent_weights(encode)
        return latent
    
if __name__ == "__main__":
    with h5py.File("C:Users\kylei\hetrec_gpt3_embed.h5", "r") as f:
        all_embed = f["Embedding"][:]
        movie_ids = f["MovieID"][:]
        titles = f["MovieTitle"][:]
        tags = f["Tags"][:]
        genres = f["Genres"][:]
    
    ae = AutoEncoder(all_embed.shape[1], reduction=0.50, dtype=torch.bfloat16)
    rand_gen = torch.Generator().manual_seed(int(b"101101010"))
    embed_tensor = torch.tensor(all_embed, device="cuda:0", dtype=torch.bfloat16)

    loader = D.DataLoader(embed_tensor,
                        generator=rand_gen,
                        batch_size=64,
                        shuffle=True,
                        )

    optimizer = torch.optim.RMSprop(ae.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

    hist = ae.fit(loader, optimizer, scheduler, criterion, epochs=50)