import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

def evaluate_model(trainer, test_dataset):
    eval_results = trainer.evaluate()
    
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    pred_labels = np.argmax(logits, axis=-1)
    true_labels = predictions.label_ids
    
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # print result
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'eval_results': eval_results
    }
