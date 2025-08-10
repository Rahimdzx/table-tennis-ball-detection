from ultralytics import YOLO

model_path = r"C:\Users\SPS\Desktop\1\runs\detect\ball_detector\weights\best.pt"
data_yaml = r"C:\Users\SPS\Desktop\1\yolo\data.yaml"

def evaluate_model():
    print("Loading model...")
    model = YOLO(model_path)
    print("Evaluating model...")
    results = model.val(data=data_yaml)
    print("Evaluation done.\n")

    print("Printing results summary:")
    results.summary()

    res_dict = results.results_dict
    print("\nDetailed results dictionary:")
    for k, v in res_dict.items():
        print(f"{k}: {v}")

    def get_metric(key_fragment):
        for k, v in res_dict.items():
            if key_fragment in k:
                return v
        return None

    precision = get_metric("precision")
    recall = get_metric("recall")
    mAP50 = get_metric("mAP50")
    mAP50_95 = get_metric("mAP50-95")
    fitness = res_dict.get("fitness", None)

    print("\nSummary metrics:")
    print(f"Precision: {precision:.4f}" if precision is not None else "Precision: N/A")
    print(f"Recall:    {recall:.4f}" if recall is not None else "Recall: N/A")
    print(f"mAP50:     {mAP50:.4f}" if mAP50 is not None else "mAP50: N/A")
    print(f"mAP50-95:  {mAP50_95:.4f}" if mAP50_95 is not None else "mAP50-95: N/A")
    print(f"Fitness:   {fitness:.4f}" if fitness is not None else "Fitness: N/A")

if __name__ == "__main__":
    evaluate_model()
