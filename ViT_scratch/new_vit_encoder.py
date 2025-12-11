import math
import torch
from torch import nn
import torch

class RGB_Depth_Agreement_Refined(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(
            self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias
        )
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
    
    def get_alpha_beta(self):
        """Sigmoid関数で0-1の範囲に制約"""
        alpha = torch.sigmoid(self.alpha_raw)
        beta = torch.sigmoid(self.beta_raw)
        return alpha, beta

    def forward(self, img, dpt, output_attentions=False):

        alpha, beta = self.get_alpha_beta()

        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv_img = self.qkv_projection(img)
        qkv_dpt = self.qkv_projection(dpt)

        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query_img, key_img, value_img = torch.chunk(qkv_img, 3, dim=-1)
        query_dpt, key_dpt, value_dpt = torch.chunk(qkv_dpt, 3, dim=-1)

        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query_img.size()
        query_img = query_img.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        key_img = key_img.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        value_img = value_img.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)

        # Calculate the attention scores
        attention_scores_img = torch.matmul(query_img, key_img.transpose(-1, -2))
        attention_scores_img = attention_scores_img / math.sqrt(self.attention_head_size)

        batch_size, sequence_length, _ = query_dpt.size()
        query_dpt = query_dpt.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        key_dpt = key_dpt.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        value_dpt = value_dpt.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)

        # Calculate the attention scores
        attention_scores_dpt = torch.matmul(query_dpt, key_dpt.transpose(-1, -2))
        attention_scores_dpt = attention_scores_dpt / math.sqrt(self.attention_head_size)

        # Softmax
        attention_probs_img = nn.functional.softmax(attention_scores_img, dim=-1)
        attention_probs_img = self.attn_dropout(attention_probs_img)

        attention_probs_dpt = nn.functional.softmax(attention_scores_dpt, dim=-1)
        attention_probs_dpt = self.attn_dropout(attention_probs_dpt)

        ## Agreementbased-Refine-Fusion
        agreement = attention_probs_img * attention_probs_dpt 
        AR_attention_probs_img = alpha * nn.functional.softmax(agreement, dim=-1) + attention_probs_img
        AR_attention_probs_dpt = beta * nn.functional.softmax(agreement, dim=-1) + attention_probs_dpt

        # Calculate the attention output
        attention_output_img = torch.matmul(AR_attention_probs_img, value_img)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output_img = (
            attention_output_img.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )

        # Calculate the attention output
        attention_output_dpt = torch.matmul(AR_attention_probs_dpt, value_dpt)
        attention_output_dpt = (
            attention_output_dpt.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )

        # print(f"attention_output:{attention_output.shape}")
        # Project the attention output back to the hidden size
        attention_output_img = self.output_projection(attention_output_img)
        attention_output_img = self.output_dropout(attention_output_img)

        # print(f"attention_output:{attention_output.shape}")
        # Project the attention output back to the hidden size
        attention_output_dpt = self.output_projection(attention_output_dpt)
        attention_output_dpt = self.output_dropout(attention_output_dpt)

        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output_img, None, attention_output_dpt, None)
        else:
            # print(f"attention_probs.shape:{attention_probs_img.shape}")
            return (attention_output_img, AR_attention_probs_img, attention_output_dpt, AR_attention_probs_dpt)
