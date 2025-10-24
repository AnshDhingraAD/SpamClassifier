import nltk

nltk.download('punkt_tab', download_dir='nltk_data')
nltk.download('stopwords', download_dir='nltk_data')
print("Download completed.")

# ===================================================================================
# This file is used to pre-download required NLTK resources (punkt tokenizer and stopwords)
# into a local folder called 'nltk_data' inside the project.
#
# Reason:
# 1. NLTK functions like `word_tokenize()` and `stopwords.words()` require additional data files.
# 2. When deploying the app on cloud platforms like Render or Heroku, these platforms
#    do NOT allow NLTK to download data at runtime.
# 3. Without pre-downloading, the app would throw a LookupError, e.g.,
#    "Resource punkt_tab not found".
# 4. By downloading locally and appending 'nltk_data' to nltk.data.path, we ensure
#    that the app can access the required resources without internet access at runtime.
#
# How it works:
# - Running this script creates a folder 'nltk_data' containing all required files.
# - The app code uses: `nltk.data.path.append("nltk_data")` to look into this folder first.
#
# This ensures:
# - The Spam Classifier works reliably both locally and on deployed servers.
# - No runtime NLTK download is needed, avoiding deployment errors.
# ===================================================================================
