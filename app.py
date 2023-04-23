from flask import Flask, render_template, request
import pickle
import os
import pandas as pd
import random
import numpy as np
# Get the path to the dataset file
# dataset_path = os.path.join(app.root_path, 'data', 'average.csv')

# Load the dataset into a Pandas dataframe
df = pd.read_csv('average.csv')
# from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index2.html')

with open('map_encoder.pkl', 'rb') as f:
    map_enc = pickle.load(f)

with open('agent_encoder.pkl', 'rb') as f:
    agent_enc = pickle.load(f)

with open('Playername_encoder.pkl', 'rb') as f:
    player_enc = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_object.pkl', 'rb') as f:
    scaler_object = pickle.load(f)

@app.route('/my-url', methods=['GET', 'POST'])
def result():
    map1=request.form['map']

    m11 = request.form['m11']
    a11 = request.form['a11']
    m21 = request.form['m21']
    a21 = request.form['a21']
    m31 = request.form['m31']
    a31 = request.form['a31']
    m41 = request.form['m41']
    a41 = request.form['a41']
    m51 = request.form['m51']
    a51 = request.form['a51']

    m12 = request.form['m12']
    a12 = request.form['a12']
    m22 = request.form['m22']
    a22 = request.form['a22']
    m32 = request.form['m32']
    a32 = request.form['a32']
    m42 = request.form['m42']
    a42 = request.form['a42']
    m52 = request.form['m52']
    a52 = request.form['a52']
    

    player_names = [m11, m21, m31, m41, m51, m12, m22, m32, m42, m52] # 1
    player_names=player_enc.transform(player_names) #after encoding

    agent_names=[a11,a21,a31,a41,a51,a12,a22,a32,a42,a52] #2
    agent_names=agent_enc.transform(agent_names) #after encdoing

    map1=map_enc.transform([map1])[0]

    avg_acs_list = [] # 3
    avg_kda_list = []   # 4

    # Loop through each player name and calculate average acs and kda
    for player_name in player_names:
        # Select rows from the dataframe corresponding to the current player
        player_df = df[df['PlayerName'] == player_name]

        # Calculate the average acs and kda for the current player
        avg_acs = player_df['ACS'].mean()
        avg_kda = player_df['KDA'].mean()

        # Append the average acs and kda to their respective lists
        avg_acs_list.append(avg_acs)
        avg_kda_list.append(avg_kda)

    print(avg_acs_list)
    game_id=random.randint(100,32873)
    members_data = []
    for i, p in enumerate(player_names):

        member_data = [game_id, map1, p, agent_names[i], avg_acs_list[i], avg_kda_list[i]]
        members_data.append(member_data)

    # print(members_data)
    members_data_scaled = scaler_object.transform(members_data)
    probabilities = []
    for i in range(members_data_scaled.shape[0]):
        probability = model.predict_proba(members_data_scaled[i:i+1])[0]
        probabilities.append(probability)
    # p1 = model.predict_proba(members_data_scaled[0])
    # p2 = model.predict_proba(members_data_scaled[1])
    # p3 = model.predict_proba(members_data_scaled[2])
    # p4 = model.predict_proba(members_data_scaled[3])
    # p5 = model.predict_proba(members_data_scaled[4])

    # p6 = model.predict_proba(members_data_scaled[5])
    # p7 = model.predict_proba(members_data_scaled[6])
    # p8 = model.predict_proba(members_data_scaled[7])
    # p9 = model.predict_proba(members_data_scaled[8])
    # p10 = model.predict_proba(members_data_scaled[9])

   
    # team1_prob_int = p1*p2*p3*p4*p5
    # team2_prob_int = p6*p7*p8*p9*p10
    # if(team2_prob_int>=team1_prob_int):
    #     winner='Team2'
    # else:
    #     winner = 'Team1'
    team1_prob_int = np.prod(probabilities[:5])
    team2_prob_int = np.prod(probabilities[5:])
    print('team1=')
    print(team1_prob_int)

    if(team1_prob_int>=team1_prob_int):
        winner='Team1'

    else :
        winner='Team1'
    return render_template('result.html', winner=winner)

if __name__ == '__main__':
    app.run(debug=True)
