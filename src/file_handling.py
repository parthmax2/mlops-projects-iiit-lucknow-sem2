import os


#-----------------------------------------
# Clean Old Ticker Files
#-----------------------------------------
def cleanup_old_ticker_files(ticker: str, data_folder='data', combine_folder='combine_data'):
    try:
        files_to_delete = [
            os.path.join(data_folder, f"{ticker}_clean.csv"),
            os.path.join(data_folder, f"{ticker}_recent.csv"),
            os.path.join(combine_folder, f"{ticker}_combine.csv")
        ]

        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Deleted: {file_path}")
            else:
                print(f"⚠️ File not found (skip): {file_path}")

    except Exception as e:
        print(f"❌ Error during cleanup: {str(e)}")
