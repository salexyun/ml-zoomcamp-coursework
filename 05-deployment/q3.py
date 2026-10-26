from common import LEAD_Q3, load_pipeline

pipeline = load_pipeline()
proba = pipeline.predict_proba([LEAD_Q3])[0, 1]
print("Q3. Conversion probability:", round(proba, 3))
