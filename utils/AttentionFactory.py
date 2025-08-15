from attention import Multi_Head_Attention, Flash_Attention, Flash_Attention_V2


class AttentionFactory:
    """
    Factory class to create attention mechanisms.
    """

    @staticmethod
    def get_attention(attention_name):
        attention_map = {
            "multihead": Multi_Head_Attention.MultiAttn,
            "flash": Flash_Attention.FlashAttention,
            "flash_v2": Flash_Attention_V2.FlashAttentionV2
        }

        return attention_map.get(attention_name.lower(), None)
