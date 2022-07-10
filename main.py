import pandas as pd
import neattext.functions as nfx
import seaborn as sns
import IPython
import eli5
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix

df = pd.read_csv("JobsDataset.csv")
df.head()
df[['Query','Description']]
df.Query.value_counts()
df['CleanDescription'] = df['Description'].apply(nfx.remove_stopwords)
df['CleanDescription'] = df['CleanDescription'].apply(nfx.remove_special_characters)
df['CleanDescription'] = df['CleanDescription'].str.lower()
df[['CleanDescription','Description']]
from sklearn.feature_extraction.text import TfidfVectorizer
Xfeatures = df['CleanDescription']
ylabels = df['Query']
Xfeatures
tfidf_vec = TfidfVectorizer()
X = tfidf_vec.fit_transform(Xfeatures)
X.todense()
#Convert to DF
df_vec = pd.DataFrame(X.todense(),columns=tfidf_vec.get_feature_names_out())
df_vec
#Dataset Spliting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,ylabels,test_size=0.3,random_state=42)

lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)

lr_model.score(x_test,y_test)

y_pred = lr_model.predict(x_test)
confusion_matrix(y_pred,y_test)
plot_confusion_matrix(lr_model,x_test,y_test,xticks_rotation=90)

example = 'Salary Range $55,789 - $89,263 annually, DOE Grade 117 Work Schedule Monday - Friday, 8am - 5pm; some eve    nings and weekends may be required College Summer Hours: Monday - Thursday, 7am-6pm Work Calendar 12 Months Job Summary Administers assigned systems by providing maintenance, security, testing, upgrades and monitoring to ensure reliability and improved system performance. Essential Functions 65% - Installs, supports, and maintains Windows servers (physical and virtual) or other computer systems and plans for and responds to and documents service outages and other problems in an Active Directory environment 15% - Installs, configures, and manages enterprise-level software (SCCM, Kaspersky, VMware, etc.) 10% - Develops high quality technical documentations for knowledge base and training tools 10% - Other duties as assigned which includes, but not limited to, scripting or light programming, project management for systems-related projects, and vendor interactions Minimum Qualifications Bachelorâ€™s Degree in information technology or directly related field and one year of experience designing, configuring and installing hardware and software, training users and maintaining servers and data communication equipment. OR An equivalent combination of education and experience sufficient to successfully perform the essential duties of the job such as those listed above, unless otherwise subject to any other requirements set forth in law or regulation. Desired Qualifications One (1) or more years managing and supporting virtual infrastructures Experience with troubleshooting and resolving server and workstation application issues in an enterprise environment Proven competency using PowerShell to develop and test scripts to automate tasks in Windows Active Directory environment, including at least three (3) of the following: User/Group management, NTFS permissions, managing file, print, SCCM, BitLocker servers. One (1) or more years of experience in an ITIL framework environment with an understanding of Change Management. ITL Foundations certification preferred. Written communication skills as expressed in a Cover Letter addressing how Minimum and Desired Qualifications are met, include specific examples. Special Working Conditions Possession of a valid State of Arizona Class D drivers license is required; must meet minimum standards regarding driving: http://www.maricopa.edu/legal/rmi/vehicle.htmrequirements Travel to campus during interview/selection process will be at candidateâ€™s own expense Will be required to travel or be assigned to all MCCCD/GCC locations May require numerous evenings or weekends May require prolonged periods of viewing a computer screen May be required to lift or carry up to 50 lbs How to Apply Applicants must submit a cover letter that details how the applicant meets minimum and desired qualifications. Applications without a cover letter will be incomplete and will not be considered. Please ensure your resume and cover letter provide the following items: Clearly illustrate how prior experience, knowledge and education meet the minimum and desired qualifications for this position. Provide employment history in a month/year format (e.g., 09/07 to 10/11) including job title, job duties, and name of employer for each position. Three professional references, preferably current and/or former supervisors. If references are not provided in resume upon application, they will be requested at time of interview. Posting Close Date Open until filled, first review of applicants begins Thursday, June 28, 2018. EEO Information Maricopa County Community College District (MCCCD) will not discriminate, nor tolerate discrimination in employment or education, against any applicant, employee, or student because of race, color, religion, sex, sexual orientation, gender identity, national origin, citizenship status (including document abuse), age, disability, veteran status or genetic information.'

def vectorize_text(text):
    my_vec = tfidf_vec.transform([text])
    return my_vec.toarray()

vectorize_text(example)
sample1 = vectorize_text(example)
lr_model.predict_proba(sample1)
lr_model.predict( sample1)

eli5.show_weights(lr_model,feature_names=tfidf_vec.get_feature_names_out())

model_file = open("lr_model_ceid.pkl","wb")
joblib.dump(lr_model,model_file)
model_file.close()

nv_model = MultinomialNB()
nv_model.fit(x_train,y_train)

nv_model.score(x_test,y_test)

y_pred2 = nv_model.predict(x_test)
confusion_matrix(y_pred2,y_test)
plot_confusion_matrix(nv_model,x_test,y_test,xticks_rotation=90)