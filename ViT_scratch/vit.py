import math
import torch
from torch import nn
import torch


def print_memory_status(label=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    print(f"[{label}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def get_list_shape(data):
    """
    再帰的にリストの形状を取得する。
    """
    if isinstance(data, list):
        if len(data) > 0:
            return [len(data)] + get_list_shape(data[0])
        else:
            return [0]  # 空リストの場合
    else:
        return []  # リストでない場合

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.num_channels_forDepth = config["num_channels_forDepth"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size

        # self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    self.num_channels,
                    self.hidden_size,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ),
                nn.Conv2d(
                    self.num_channels_forDepth,
                    self.hidden_size,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ),
                nn.Conv2d(
                    (self.num_channels + self.num_channels_forDepth),
                    self.hidden_size,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ),
            ]
        )

    def forward(self, x, Isdepth=False):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        # x = self.projection(x)
        projection_layer = self.projections[1] if Isdepth else self.projections[0]

        # Apply the selected projection layer
        x = projection_layer(x)  ### x is embedding

        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, Isdepth=False):
        # print(f"pe:  {self.position_embeddings.shape}")
        x = self.patch_embeddings(x, Isdepth)
        # print(f"x:  {x.shape}")
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
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
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias,
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs], dim=-1
        )
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs], dim=1
            )
               
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
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

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.attention_head_size,
        ).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V

        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )
        # print(f"attention_output:{attention_output.shape}")
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            # print(f"attention_probs.shape:{attention_probs.shape}")
            return (attention_output, attention_probs)

class RGB_Depth_CrossMultiHeadAttention(nn.Module):
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
        self.alpha = config["alpha"]
        self.beta = config["beta"]

    def forward(self, img, dpt, output_attentions=False):
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
        # softmax(Q*K.T/sqrt(head_size))*V

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
        attention_scores_dpt = torch.matmul(query_dpt, key_dpt.transpose(-1, -2))
        attention_scores_dpt = attention_scores_dpt / math.sqrt(self.attention_head_size)

        ## cross attention
        shared_attention_scores_img = attention_scores_img + self.alpha*attention_scores_dpt
        shared_attention_scores_dpt = attention_scores_dpt + self.beta*attention_scores_img


        attention_probs_img = nn.functional.softmax(shared_attention_scores_img, dim=-1)
        attention_probs_img = self.attn_dropout(attention_probs_img)
        # Calculate the attention output
        attention_output_img = torch.matmul(attention_probs_img, value_img)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output_img = (
            attention_output_img.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.all_head_size)
        )

        attention_probs_dpt = nn.functional.softmax(shared_attention_scores_dpt, dim=-1)
        attention_probs_dpt = self.attn_dropout(attention_probs_dpt)
        # Calculate the attention output
        attention_output_dpt = torch.matmul(attention_probs_dpt, value_dpt)
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
            # print(f"attention_probs.shape:{attention_probs.shape}")
            return (attention_output_img, attention_probs_img, attention_output_dpt, attention_probs_dpt)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        self.use_method1 = config.get("use_method1", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        elif self.use_method1: 
            self.attention = RGB_Depth_CrossMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        # print(f"Block.forward: input output_attentions={output_attentions}, type={type(output_attentions)}")
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)

class Block_RGB_Depth(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        self.use_method1 = config.get("use_method1", False)
        if self.use_method1: 
            self.attention = RGB_Depth_CrossMultiHeadAttention(config)
        elif self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, img, dpt, output_attentions=False):
        # Self-attention
        # print(f"Block.forward: input output_attentions={output_attentions}, type={type(output_attentions)}")
        attention_output_img, attention_probs_img, attention_output_dpt, attention_probs_dpt = self.attention(
            self.layernorm_1(img), self.layernorm_1(dpt), output_attentions=output_attentions
        )
        # Skip connection
        img = img + attention_output_img
        dpt = dpt + attention_output_dpt
        # Feed-forward network
        mlp_output_img = self.mlp(self.layernorm_2(img))
        mlp_output_dpt = self.mlp(self.layernorm_2(dpt))
        # Skip connection
        img = img + mlp_output_img
        dpt = dpt + mlp_output_dpt
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (img, None, dpt, None)
        else:
            return (img, attention_probs_img, dpt, attention_probs_dpt)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        # num_hidden_layers is encoder blocks number
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        # print(f"Encoder start: output_attentions={output_attentions}, type={type(output_attentions)}")
        for block in self.blocks:

            x, attention_probs = block(x, output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        
        if not output_attentions:
            return (x, None)
        else:
            # all_attention is all block attention in entire Encoder
            return (x, all_attentions)

class Encoder_RGB_Depth(nn.Module):
    """
    The transformer encoder module specialized for multimordal
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        # num_hidden_layers is encoder blocks number
        for _ in range(config["num_hidden_layers"]):
            block = Block_RGB_Depth(config)
            self.blocks.append(block)

    def forward(self, img, dpt, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions_img = []
        all_attentions_dpt = []
        # print(f"Encoder start: output_attentions={output_attentions}, type={type(output_attentions)}")
        for block in self.blocks:

            img, attention_probs_img, dpt, attention_probs_dpt = block(img, dpt, output_attentions=output_attentions)

            if output_attentions:
                all_attentions_img.append(attention_probs_img)
                all_attentions_dpt.append(attention_probs_dpt)
        # Return the encoder's output and the attention probabilities (optional)
        
        if not output_attentions:
            return (img, None, dpt, None)
        else:
            # all_attention is all block attention in entire Encoder
            return (img, all_attentions_img, dpt, all_attentions_dpt)



class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, attentions_choice=False):
        # Calculate the embedding output
        embedding_output_image = self.embedding(x, Isdepth=False)

        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(
            embedding_output_image, output_attentions=attentions_choice
        )
        
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)

        if not attentions_choice:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)


class EarlyFusion(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, img, dpt, output_attentions=False, Isdepth=False):
        # Calculate the embedding output
        embedding_output_image = self.embedding(img, Isdepth=Isdepth)
        # print(f"[test] emd_image: {embedding_output_image.shape}")
        embedding_output_depth = self.embedding(dpt, Isdepth=True)
        # print(f"[test] emd_depth: {embedding_output_depth.shape}")
        embedding_output_fusion = torch.cat(
            (embedding_output_image, embedding_output_depth), dim=1
        )
        # print(f"[test] emd_fusion: {embedding_output_fusion.shape}")

        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(
            embedding_output_fusion, output_attentions=output_attentions
        )
        # print(f"[test] encoder_output: {encoder_output.shape}")
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])

        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)

class LateFusion(nn.Module):
    """
    The ViT model for classification using Late Fusion.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.use_method1 = config.get("use_method1", False)
        # Create the embedding module
        self.embedding = Embeddings(config)
        
        # Create the transformer encoder module (shared for simplicity)
        self.encoder_rgb = Encoder(config)
        self.encoder_depth = Encoder(config)

        ### ----- use common Encoder ------
        self.encoder_rgb_depth = Encoder_RGB_Depth(config)

        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, img, dpt, attentions_choice=False):
        # Calculate the embedding output for RGB
        embedding_output_rgb = self.embedding(img, Isdepth=False)
        # print(f"here::{embedding_output_rgb.shape}")
        # Calculate the embedding output for Depth
        embedding_output_depth = self.embedding(dpt, Isdepth=True)
        # print(f"here{attentions_choice}")
        if self.use_method1:
            # Pass through separate encoders
            encoder_output_rgb, all_attentions_rgb, encoder_output_depth, all_attentions_depth = self.encoder_rgb_depth(
                embedding_output_rgb, embedding_output_depth, output_attentions=attentions_choice
            )

        else: 
            # Pass through separate encoders
            encoder_output_rgb, all_attentions_rgb = self.encoder_rgb(
                embedding_output_rgb, output_attentions=attentions_choice
            )
            encoder_output_depth, all_attentions_depth = self.encoder_depth(
                embedding_output_depth, output_attentions=attentions_choice
            )
            
        # Fusion (simple addition here, but can be concatenation or weighted sum)
        fusion_output = encoder_output_rgb + encoder_output_depth
        
        # Classification using [CLS] token
        logits = self.classifier(fusion_output[:, 0, :])

        # print(f"here:attention_rgb{type(all_attentions_rgb)}

        if not attentions_choice:
            return (logits, None)
        else:
            # print(f"all_attentions_rgb shape: {all_attentions_rgb[2].shape}")
            # print(aaa)
            # print(f"all_attentions_depth shape: {get_list_shape(all_attentions_depth)}")
            return (logits, all_attentions_rgb, all_attentions_depth)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
