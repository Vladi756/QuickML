# QuickML Documentation

## 1. Installing VMWare

VMWare, or an equivalent (VirtualBox, etc.) needs to be installed to be able to run virtual envrionments.


## 2. Data Pre-Processing 

The first step to creating a machine learning model is preparing the data to be fed into it by pre-processing. The data needs to be pre-processed and the following steps followed:

1. Acquire the Dataset 
2. Import Necessary Libraries 
3. Import the Dataset
4. Handling Missing Values
5. Encoding Categorical Data
6. Splitting into Training and Test Set
7. Feature Scaling


```python
# Importing All Libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```


```python
# Mapping independent, dependent, categorical and missing data
# to begin data pre-processing.
var_map = {
    "independent" : ["R&D Spend", "Administration","Marketing Spend", "State"],
    "dependent" : ["Profit"],
    "categorical" : ["State"],
    "missing": ["Marketing Spend"]
}
```


```python
# Defining Function 
def dataPreProcess(dataSet, varMap):
    # Obtaining Data Set
    data_root = pd.read_csv(dataSet)
    data = data_root.copy()

    # Splitting Dependent & Independent Variables
    X = data[varMap['independent']]  
    y = data[varMap['dependent']]

    # Removing any missing data
    imputer = SimpleImputer(missing_values=np.nan , strategy='mean')
    imputer = imputer.fit(X[varMap['missing']])
    X[varMap['missing']] =imputer.transform(X[varMap['missing']])

    # Encoding Categorical Variables
    le = LabelEncoder()
    X[varMap['categorical']]= pd.DataFrame(le.fit_transform(X[varMap['categorical']]))
    col_tans = make_column_transformer( 
                         (OneHotEncoder(), 
                         varMap['categorical']))
    Xtemp2 = col_tans.fit_transform(X[varMap['categorical']])
    # Splitting Into Train and Test Set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 0)

    # Feature Scaling
    scale_X = StandardScaler()
    X_train.iloc[: , :] = scale_X.fit_transform(X_train.iloc[: , :])
    X_test.iloc[: , :] = scale_X.fit_transform(X_test.iloc[: , :])

    # Returns a dictionary of pre-processed data
    return(
        {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_train
        }
    )
```

The data processing function is responsible for taking a dataset and a mapping of dependent, independent, missing and categorical data. The dataset is split into the dependent and independent data, the missing data is taken care of, and the categorical data is encoded.

Finally, the data is split into the test and train and it is feature scaled. The function returns a dictionary of the train and test matrices and vectors ready for a machine learning model to be fitted on. 

### 2.1. Dynamic Table Creation

Once the Algorithm of choice is selected, an HTML table is dynamically created with the column names:
1. Independent
2. Dependent
3. Categorical 
As well as dynamically created row names which correspond to the attributes in the inputted data set. This was done using Flask and Jinja. 

Additionally, the file the user submits is saved to a specific folder, effectively keeping a reference to this file to be used later on. 


```python
# Invoked when user submits file - 
# creates HTML table with attributes of file
@views.route('/', methods=['POST'])
def upload_file():
    
    global filename
    file = request.files['file']
    
    # Saves the file so it can be accessed later on.
    dataSet = pd.read_csv(file)
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    filename = file.filename

    return render_template('home.html', attributes = list(dataSet.columns))
```


```python
 {% for i in attributes: %}
                <tr>
                    <th class="rt" value='{{i}}'>{{i}}</th>
                    <td><input class='rd' type="radio" name='test_{{attributes.index(i)}}' 
                               value="Ind"></td>
                    <td><input class='rd dep' type="radio" name='test_{{attributes.index(i)}}'
                               value="Dep"></td>
                    <td><input class='rd' type="checkbox" value="Cat"></td>
                </tr>
{% endfor %}
<!-- Using Jinja python expressions can be written in html. 
    Table created dynamically using for loop. -->
```

### 2.2. Dynamic Creation of Variable Mapping

The input of the radio buttons and checkboxes on the dynamically created table are used to create the mapping of attributes. Namely, the user selects which attributes are dependent, independent, and which are categorical. 


```python
function makeVarMap() {

  const radioButtons = document.querySelectorAll('.rd');  // All radio buttons
  const headers = document.querySelectorAll('th.rt');     // Headers
  // Hard coded keys as they never change regardless of use case. 
  var varMap = {
    'Independent': [],
    'Dependent': [],
    'Categorical': []
  }

  var head = [];
  for (let i = 0; i < headers.length; ++i) {
    head[i] = headers[i].textContent;
  }
  var j = 0;
  // Loops through radio buttons 
  for (let x = 0; x < radioButtons.length; x++) {
    if (radioButtons[x].checked && radioButtons[x].value == 'Ind') {
      varMap['Independent'].push(head[j]);
      j++;
    }
    if (radioButtons[x].checked && radioButtons[x].value == 'Dep') {
      varMap['Dependent'].push(head[j]);
      j++;
    }
    if (radioButtons[x].checked && radioButtons[x].value == 'Cat') {
      // Decrements becase a categorical variable is always ALSO ind or dep.   
      j--;
      varMap['Categorical'].push(head[j]);
      j++;
      // Increments so order is not messed up.
    }
  }
  console.log(varMap);
}
```

There is also some checkbox logic implemented such that a variable cannot be both independent and dependent and that there can only ever be one dependent variable in any inputted dataset. This was done in jQuery.


```python
$(document).ready(function () {
  $('input.dep:radio').change(function() {
      // When any radio button on the page is selected,
      // then deselect all other radio buttons.
      $('input.dep:radio:checked').not(this).prop('checked', false);
  });
})
```

### 2.3. Passing Variable Mapping to be Pre Processed

The variable mapping is created in JavaScript dnyamically using the users input. It is then passed to the python backend using AJAX:


```python
$.ajax({
    url: '/dataPreProcessing',
    type: "POST",
    contentType: "application/json", 
    data: JSON.stringify(s)
  }).done(function(result){     // on success get the return object from server
    console.log(result)     // see it in the console to test its working 
})
```

### 2.4. Pre Processing the Data 

Once the variable mapping is created in the JavaScript, it is passed into the flask backend which takes the original file the user submitted, as well as the newly created variable mapping, passing both of them as arguments to the data pre-processing function. 


```python
# Invoked when user submits variable mapping 
@views.route('/dataPreProcessing', methods=['POST'])
def dataPre():
    # result is the variable mapping in a JSON format
    result =  request.get_json()

    # Dataset and variable mapping to be passed into the data
    # pre-processing function
    varMap = json.loads(result)
    file = os.path.join(UPLOAD_FOLDER, filename)

    table = DPP.dataPreProcess(file, varMap)

    # Getting the individual components of pre processed data 
    # to keep a reference to them for when they need to be passed 
    # in to the selected algorithm.
    xTest = pd.DataFrame(table['X_test'])
    xTrain = pd.DataFrame(table['X_train'])
    yTest = pd.DataFrame(table['y_test'])
    yTrain = pd.DataFrame(table['y_train'])   

    # Creating variables to store file names and locations for pre 
    # processed data locations
    fN_xT = '/home/user/Documents/git/QuickML/pre_processed_data/xTest'
    fN_xTr = '/home/user/Documents/git/QuickML/pre_processed_data/xTrain'
    fN_yT = '/home/user/Documents/git/QuickML/pre_processed_data/yTest'
    fN_yTr = '/home/user/Documents/git/QuickML/pre_processed_data/yTrain'

    # pd.to_csv creates the file if it does not exist, but it does not 
    # create any non existent directories. The pre_processed_data directory 
    # already exists, pd.to_csv <i>creates</i> the files and populates them 
    # with the contents of their respective components. 
    xTest.to_csv(fN_xT)
    xTrain.to_csv(fN_xTr)
    yTest.to_csv(fN_yT)
    yTrain.to_csv(fN_yTr)

    # Getting the file out of the whole path and converting it to a dataframe.
    dF = pd.read_csv(file.split('/')[-1])
    
    # Columns still hard coded! Fix before deploying to production. 
    col = dF.columns

    # return formattes string which contains HTML and HTML tables using 
    # the 'tabulate' module
    return (f'''
            <h2 style="text-align:center">Scroll to Preview your Pre-Processed Data!</h2>
            <hr>
            <div>
                <h3 style="text-align:left"> X train </h3> 
                <h3 style="text-align:right; margin-top:-40px"> Y train </h3> <hr><br>
                <div class="container" style="display:flex; width=70%">
                    {tabulate(table['X_train'], tablefmt='html', headers = col)}
                    {tabulate(table['y_train'], tablefmt='html', headers = col[4:])}
                </div>
                <hr>
                <h3 style="text-align:left"> X test </h3> 
                <h3 style="text-align:right; margin-top:-40px"> Y test </h3> <hr><br>
                <div class="container" style="display:flex; width=70%">
                    {tabulate(table['X_test'], tablefmt='html', headers = col)}
                    {tabulate(table['y_test'], tablefmt='html', headers = col[4:])}
                </div>
            </div>
    ''' )
```

It also writed 4 csv files each containing one of the components of the pre-processed data:
1. X_train
2. y_train 

These are datasets which will be used to train the Machine Learning/Deep Learning model. 

3. X_test
4. y_train

These are the datasets which are given to the ML/DL model to test it's accruacy. Based on these results, the confusion matrix is created. 

It is also important to keep a reference to the users choice of algorithm so that the correct one is invoked. This is sent from the JavaScript to the Flask backend, which then writes it to a text file: 


```python
// JavaScript Code 
document.querySelectorAll('.choice').forEach(item => {
    item.addEventListener("click", event => {
      var value = item.value;
        
      // Users choice 
      let option = value
      console.log(option)
      // Open and send information to Flask
      const request = new XMLHttpRequest()
      request.open('POST', `/ProcessOption/${JSON.stringify(option)}`)
      request.send();
    })
  })
```


```python
# Flask Backend Code 
@views.route('/ProcessOption/<string:option>', methods=['POST'])
def SaveOption(option):
    sel = json.loads(option)

    with open("choice.txt", "w") as fo:
        fo.write(sel)
    return 1
```

This marks the end of the data pre-processing section. Now the algorithms can be analyzed.

## 3. Regression 

### 3.1 Simple Linear Regression 

A Simple Linear Regression is a machine learning model used on data sets with 2 columns, one independent and one dependent variable. Below is the code for the algorithm: 


```python
# SIMPLE LINEAR REGRESSION
def simpleLinearRegression(Xtest, Xtrain, Ytest, Ytrain, dataSet):
    """
    Takes the train and test split of the dataset, as well as name
    of the uploaded dataSet. Fits a regressor and plots a simple
    linear regression on the dataset. Saves the figure and returns path
    to saved figure as jpg.
    """
    
    regressor = LinearRegression()
    regressor.fit(Xtrain, Ytrain)   

    plt.title(f'Linear Regression for Dataset: {dataSet}')

    plt.scatter(Xtest[:,:].transpose()[1:,:].tolist()[0], 
                Ytest[:,:].transpose()[1:,:].tolist()[0], 
                color='blue',   
                label='Test Samples')
    
    plt.scatter(Xtrain[:,:].transpose()[1:,:].tolist()[0],
                Ytrain[:,:].transpose()[1:,:].tolist()[0],
                color='red', 
                label = 'Train Samples')
  
    XTest_Plot = Xtest[:,:].transpose()[1:,:].tolist()[0]
    (Ytrain[:,:].transpose()[1:,:].tolist()[0])

    Ytrain_temp = regressor.predict(Xtest) 

    YTrain_Hat_Plot = Ytrain_temp.transpose()[1:,:].tolist()[0]

    plt.plot(sorted(XTest_Plot),
             sorted(YTrain_Hat_Plot),
             label='Regression line')

    plt.legend()
    plt.grid()

    filename = f'{random.randint(100,999)}'
    plt.savefig(f'/home/user/Documents/git/QuickML/webapp/static/{filename}.jpg')

    x = f'/home/user/Documents/git/QuickML/webapp/static/{filename}.jpg'

    return x 

    # The function returns the path to the figure, which can then be viewed
    # by the user.
```

### 3.2 Multiple Linear Regression


