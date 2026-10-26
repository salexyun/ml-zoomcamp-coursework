from common import LEAD_Q4, local_server, predict

with local_server():
    result = predict(LEAD_Q4)

print("Response:", result)
print("Q4. Conversion probability:", round(result["conversion_probability"], 3))
