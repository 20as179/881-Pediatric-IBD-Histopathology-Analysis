
import requests

id_array = [55784, 55790, 55812, 55824, 55798, 55829, 55782, 55793, 55794, 55799, 55803, 55805, 55830, 55786, 55822, 57185, 55785, 55788, 55789, 55797, 55801, 55804, 55819, 55821, ]  # Add more IDs as needed
for numeric_id in id_array:
    url = f"https://cildata.crbs.ucsd.edu/media/images/{numeric_id}/{numeric_id}.zip"
    print(f"Downloading {numeric_id}.zip")
    response = requests.get(url)
    with open(f"{numeric_id}.zip", "wb") as f:
        f.write(response.content)