import requests
import pyDOE

if __name__ == "__main__":
    X = pyDOE.lhs(93, samples=32, criterion='maximin')
    url = 'http://localhost:5000/predict'
    for ii in range(30):
        data = {'input': X[ii:ii+2].tolist(),
                'para':'Mass_flow',
                }
        response = requests.post(url, json=data)
        print(response.json())