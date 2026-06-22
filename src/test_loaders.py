from data_loaders import load_adult_dataset, load_sms_dataset

adult = load_adult_dataset()
sms = load_sms_dataset()

print("Adult dataset shape:", adult.shape)
print("SMS dataset shape:", sms.shape)