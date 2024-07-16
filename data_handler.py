import pandas as pd

def load_data(filepath): # This Function loads the data from the CSV file into a DataFrame
    return pd.read_csv(filepath)

def get_pokemon_by_name(data, name): # Returns data of Pokemon based on name
    return data[data['name'].str.lower() == name.lower()]

if __name__ == "__main__": # Testing output
    filepath = "data/PokeInfo.csv"
    data = load_data(filepath)
    print(get_pokemon_by_name(data, "Mew"))
    print(data.index)
    print(len(data['name']))
    print("Index range in data:", data.index.min(), "to", data.index.max())