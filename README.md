# Next step for the CSLS model:
- invesitgate whether we can get a warped representation

ToDo:
- replace the hidden layer of the feedfroward cortical system with an LSTM which could be PFC
- Then feed the last hidden rep. in the final step of the LSTM to the MLP for making a decision
- Concatante F1, F2, and axis (i.e., context) in a way that LSTM would recognize it at 3 time steps
- So LSTM would see axis first, then F1, then F2 and then make a decision
- Then we can look at the F1 and F2 hidden representation and calculate the distance between them over the course of training
- If there is any warped representation, then the Euclidean distance between F1 and F2 would increase if it is a congruent trials compare to incongrunet

