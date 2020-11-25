# social media prediction

## run the app
python app.py


## rules
- Teams of 3-4
- Try to achieve an MSLE of at least 1.5
- You are NOT allowed to use additional external data
- You are NOT obliged to use all the features present in the dataset.
- You can only use features derived from the given features.
- 2 submissions max per day
- Please checkpoint your code for each submission you make, so that it is accessible during the code submission.
- Clean and preprocess the data as you deem appropriate before trying to fit a model.
- Apply yourself!
- Have fun!


## download the data
if you are using Linux.
1. pip install kaggle
2. cd ~/.kaggle
3. homepage www.kaggle.com -> Your Account -> Create New API token
4. mv ~/Downloads/kaggle.json ./
5. chmod 600 ./kaggle.json
6. cd ~/Downloads
7. kaggle competitions download -c ift6758-a20
8. unzip under /data in the cloned repository

## data fields
- Id - Anonymous unique alphanumeric id for each user
- User Name - The screen name of the user
- Personal URL - The link to the personal webpage of the user, if provided
- Profile Cover Image Status - Whether the user has used a profile 'cover image' (additional image) on the profile
- Profile Verification Status - The status of the user's profile verification (verified, nt verified, pending, etc.) (validating if the profile is of a genuine user)
- Profile Text Color - The hex code (RRGGBB) of the user profile's text color
- Profile Page Color - The hex code (RRGGBB) of the user profile's page color
- Profile Theme Color - The hex code (RRGGBB) of the user profile's theme color
- Is Profile View Size Customized? - The boolean value if the user has customized the view size of the profile
- UTC Offset The difference of the local time (zone) of the user with respect to the UTC specified in seconds along with a sign +/- to denote the difference
- Location - The text provided by the user to indicate his/her location
- Location Public Visibility - The status of the user's location being shared publicly
- User Language - The abbreviation of the language set by the user to use the profile specified according to (ISO 639-1)[https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes]
- Profile Creation Timestamp - The string indicating the timestamp of the user's profile creation date and time specified as an ISO string
- User Time Zone - The name of the city/time zone of the user
- Num of Followers - The number of followers the user's profile
- Num of People Following - The number of profiles followed by the user
- Num of Status Updates
- Num of Direct Messages
- Profile Category - The category of the user profile as identified by the user - eg. `business`, `government`, etc.
- Avg Daily Profile Visit Duration in seconds - The average values of the number of seconds visitors spend on the user's profile page.
- Avg Daily Profile Clicks - The average number of mouse clicks that visitors of the user's profile make - whether it is for reading threads or pressing other buttons.
- Profile Image - The filename of the PNG image corresponding to the user's profile image used in the simulation. These filenames point to the image file name in the train_profile_images.zip and test_profile_images.zip files.
- Num of Profile Likes - The number of profile 'likes' received by the user from visitors/followers 
