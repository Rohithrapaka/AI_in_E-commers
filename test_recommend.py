from recommend import hybrid_recommend

user_id = "U10"
results = hybrid_recommend(user_id, top_n=5)
print(f"\nTop 5 Product Recommendations for {user_id}:\n")
print(results)
