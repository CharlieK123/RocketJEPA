def forward_w(self, state_t, state_tk):
    # use current state to predict future latent state

    latent_t = self.encoder(state_t)
    latent_tk_hat = self.predictor(latent_t)

    # get the true latent future state with stop gradient target encoder
    with torch.no_grad():
        latent_tk = self.target_encoder(state_tk)

    return latent_tk_hat, latent_tk
