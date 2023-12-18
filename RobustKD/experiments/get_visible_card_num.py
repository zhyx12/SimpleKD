import os
import torch


def get_visible_card_num():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        cards_num = cards.count(',') + 1
        real_num = torch.cuda.device_count()
        # assert cards_num <= real_num, 'CUDA_VISIBLE_DEVICES {} should be lower than real number {}'.format(cards_num,
        #                                                                                                    real_num)
        return cards_num
    else:
        return torch.cuda.device_count()


if __name__ == "__main__":
    print(get_visible_card_num())
