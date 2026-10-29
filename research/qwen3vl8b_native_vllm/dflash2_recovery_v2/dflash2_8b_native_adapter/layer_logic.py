"""DFlash2 four-convolution layer with official deferred residual convention."""


def decoder_forward(self, positions, hidden_states, residual):
    if residual is not None:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    else:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    hidden_states, output_kernel = self.attention_conv.prepare(hidden_states)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    hidden_states = self.attention_conv.finish(hidden_states, output_kernel)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states, output_kernel = self.mlp_conv.prepare(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.mlp_conv.finish(hidden_states, output_kernel)
    return hidden_states, residual
