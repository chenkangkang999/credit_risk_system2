import hashlib


class ABTestRouter:
    def __init__(self, model_a, model_b, ratio=0.5):
        if not 0 <= ratio <= 1:
            raise ValueError("A 组比例必须在 0 到 1 之间")
        self.model_a = model_a
        self.model_b = model_b
        self.ratio = ratio

    def group(self, user_id):
        if user_id is None or not str(user_id).strip():
            raise ValueError("请输入申请编号")
        bucket = int(hashlib.sha256(str(user_id).encode()).hexdigest()[:8], 16) / 2**32
        return "A" if bucket < self.ratio else "B"

    def route(self, user_id, features):
        group = self.group(user_id)
        model = self.model_a if group == "A" else self.model_b
        return group, model.predict_proba(features)
