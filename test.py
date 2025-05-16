from backend.api.fact_check import fact_check_claim


if __name__ == "__main__":
    sample_text = "Modi is a great leader who changed India forever!"
    result = fact_check_claim(sample_text)
    print(result)
