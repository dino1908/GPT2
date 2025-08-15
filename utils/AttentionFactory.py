from attention import Multi_Head_Attention, Flash_Attention


class AttentionFactory:
    """
    Factory class to create attention mechanisms.
    """

    @staticmethod
    def get_attention(attention_name):
        attention_map = {
            "multihead": Multi_Head_Attention.MultiAttn,
            "flash": Flash_Attention.FlashAttention
        }

        return attention_map.get(attention_name.lower(), None)
