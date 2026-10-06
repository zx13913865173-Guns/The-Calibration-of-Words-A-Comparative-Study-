import json

def intent_retention(ratings):
    """
    ratings: list of 0/1
    """
    return sum(ratings) / len(ratings)

def compute_ctme(delta_clip, retention):
    return abs(delta_clip) + (1.0 - retention)

if __name__ == "__main__":
    # 读取盲评数据
    pass
