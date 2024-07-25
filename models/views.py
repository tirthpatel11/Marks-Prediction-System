import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
import warnings
from .forms import StudentDataForm
from django.http import JsonResponse
import io
import urllib, base64
from django.shortcuts import render
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore")

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def your_view_function(request):
    df = pd.read_csv('models/StudentsPerformance.csv')
    X = df.drop(columns=['math score'], axis=1)
    y = df['math score']

    num_features = X.select_dtypes(exclude="object").columns
    cat_features = X.select_dtypes(include="object").columns

    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),
        ]
    )

    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
    }

    model_results = []
    r2_list = [] 

    for model_name, model in models.items():
        model.fit(X_train, y_train)  

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
        model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

        model_results.append({
            'model_name': model_name,
            'train_performance': {
                'rmse': model_train_rmse,
                'mse': model_train_mse,
                'mae': model_train_mae,
                'r2': model_train_r2
            },
            'test_performance': {
                'rmse': model_test_rmse,
                'mse': model_test_mse,
                'mae': model_test_mae,
                'r2': model_test_r2
            }
        })

        linear_regression_model = LinearRegression()
        linear_regression_model.fit(X_train, y_train)
        r2_list.append(model_test_r2)
        model_list = [model_name for model_name in models.keys()]
        r2_scores = pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"], ascending=False)

        form = StudentDataForm()
        predictions = {}
    if request.method == 'POST':
        form = StudentDataForm(request.POST)
        if form.is_valid():
            new_student_data = form.cleaned_data
            transformed_data = preprocessor.transform(pd.DataFrame(new_student_data, index=[0]))
            linear_regression_predictions = linear_regression_model.predict(transformed_data)
            predictions = {
                'Linear Regression': linear_regression_predictions[0]
            }
            return JsonResponse({'predictions': predictions})
        else:
            return JsonResponse({'error': 'Form is not valid'})
    else:
        # If it's a GET request, render the empty form
        form = StudentDataForm()

    df = pd.read_csv('models/StudentsPerformance.csv')
    df['total score'] = df['math score'] + df['reading_score'] + df['writing_score']
    df['average'] = df['total score']/3
    # Plotting code for each plot


    # Plot 1
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    sns.set(style="whitegrid")

    sns.histplot(data=df, x='average', kde=True, ax=axs[0], palette="Set1", common_norm=False)
    axs[0].set_xlabel('Average', fontsize=14)
    axs[0].set_ylabel('Frequency', fontsize=14)
    axs[0].set_title('Histogram of Average', fontsize=16)

    sns.histplot(data=df, x='average', kde=True, hue='gender', ax=axs[1], palette="Set1", common_norm=False)
    axs[1].set_xlabel('Average', fontsize=14)
    axs[1].set_ylabel('Frequency', fontsize=14)
    axs[1].set_title('Histogram of Average by Gender', fontsize=16)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot1_image = "data:image/png;base64,{}".format(graphic)

    # Plot 2
    plt.figure(figsize=(20, 10))
    plt.subplot(141)
    sns.histplot(data=df, x='average', kde=True, hue='lunch', palette="Set2")
    plt.xlabel('Average', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Average', fontsize=16)

    plt.subplot(142)
    sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='lunch', palette="Set2")
    plt.xlabel('Average', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Average (Female)', fontsize=16)

    plt.subplot(143)
    sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='lunch', palette="Set2")
    plt.xlabel('Average', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Average (Male)', fontsize=16)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot2_image = "data:image/png;base64,{}".format(graphic)

    # Plot 3
    plt.figure(figsize=(20, 10))
    plt.subplot(141)
    sns.histplot(data=df, x='average', kde=True, hue='race_ethnicity', palette="Set3")
    plt.xlabel('Average', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Average', fontsize=16)

    plt.subplot(142)
    sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='race_ethnicity', palette="Set3")
    plt.xlabel('Average', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Average (Female)', fontsize=16)

    plt.subplot(143)
    sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='race_ethnicity', palette="Set3")
    plt.xlabel('Average', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Average (Male)', fontsize=16)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot3_image = "data:image/png;base64,{}".format(graphic)

    # Plot 4
    plt.figure(figsize=(30, 12))
    plt.subplot(1, 5, 1)
    size = df['gender'].value_counts()
    labels = 'Female', 'Male'
    color = ['red', 'green']

    plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 20})
    plt.title('Gender', fontsize=20)
    plt.axis('off')

    plt.subplot(1, 5, 2)
    size = df['race_ethnicity'].value_counts()
    labels = 'Group C', 'Group D', 'Group B', 'Group E', 'Group A'
    color = ['red', 'green', 'blue', 'cyan', 'orange']

    plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 20})
    plt.title('Race/Ethnicity', fontsize=20)
    plt.axis('off')

    plt.subplot(1, 5, 3)
    size = df['lunch'].value_counts()
    labels = 'Standard', 'Free'
    color = ['red', 'green']

    plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 20})
    plt.title('Lunch', fontsize=20)
    plt.axis('off')

    plt.subplot(1, 5, 4)
    size = df['test_prep_course'].value_counts()
    labels = 'None', 'Completed'
    color = ['red', 'green']

    plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 20})
    plt.title('Test Course', fontsize=20)
    plt.axis('off')

    plt.subplot(1, 5, 5)
    size = df['parental_education'].value_counts()
    labels = 'Some College', "Associate's Degree", 'High School', 'Some High School', "Bachelor's Degree", "Master's Degree"
    color = ['red', 'green', 'blue', 'cyan', 'orange', 'grey']

    plt.pie(size, colors=color, labels=labels, autopct='%.2f%%', textprops={'fontsize': 20})
    plt.title('Parental Education', fontsize=20)
    plt.axis('off')

    plt.tight_layout()
    plt.grid()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot4_image = "data:image/png;base64,{}".format(graphic)

    # Plot 5
    f, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.countplot(x=df['gender'], data=df, palette='bright', ax=ax[0], saturation=0.95)
    for container in ax[0].containers:
        ax[0].bar_label(container, color='black', size=14)

    ax[1].pie(x=df['gender'].value_counts(), labels=['Male', 'Female'], explode=[0, 0.1], autopct='%1.1f%%',
            shadow=True, colors=['#ff4d4d', '#ff8000'], textprops={'fontsize': 14})

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot5_image = "data:image/png;base64,{}".format(graphic)

    # Plot 6
    Group_data2 = df.groupby('race_ethnicity')
    f, ax = plt.subplots(1, 3, figsize=(20, 8))
    sns.barplot(x=Group_data2['math score'].mean().index, y=Group_data2['math score'].mean().values, palette='mako',
                ax=ax[0])
    ax[0].set_title('Math score', color='#005ce6', size=16)
    for container in ax[0].containers:
        ax[0].bar_label(container, color='black', size=14)

    sns.barplot(x=Group_data2['reading_score'].mean().index, y=Group_data2['reading_score'].mean().values,
                palette='flare', ax=ax[1])
    ax[1].set_title('Reading score', color='#005ce6', size=16)
    for container in ax[1].containers:
        ax[1].bar_label(container, color='black', size=14)

    sns.barplot(x=Group_data2['writing_score'].mean().index, y=Group_data2['writing_score'].mean().values,
                palette='coolwarm', ax=ax[2])
    ax[2].set_title('Writing score', color='#005ce6', size=16)
    for container in ax[2].containers:
        ax[2].bar_label(container, color='black', size=14)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot6_image = "data:image/png;base64,{}".format(graphic)

    # Plot 7
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.barplot(x=df['lunch'], y=df['math score'], hue=df['test_prep_course'], palette="Set3")
    plt.subplot(2, 2, 2)
    sns.barplot(x=df['lunch'], y=df['reading_score'], hue=df['test_prep_course'], palette="Set3")
    plt.subplot(2, 2, 3)
    sns.barplot(x=df['lunch'], y=df['writing_score'], hue=df['test_prep_course'], palette="Set3")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot7_image = "data:image/png;base64,{}".format(graphic)

    # Plot 8
    sns.set(style="white")
    sns.pairplot(df, hue='gender', palette="Set1")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot8_image = "data:image/png;base64,{}".format(graphic)

    #plot 9
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    sns.set(style="whitegrid")

    sns.histplot(data=df, x='average', kde=True, hue='parental_education', ax=axs[0])
    axs[0].set_xlabel('Average', fontsize=14)
    axs[0].set_ylabel('Frequency', fontsize=14)
    axs[0].set_title('Histogram of Average', fontsize=16)

    sns.histplot(data=df[df.gender == 'male'], x='average', kde=True, hue='parental_education', ax=axs[1])
    axs[1].set_xlabel('Average', fontsize=14)
    axs[1].set_ylabel('Frequency', fontsize=14)
    axs[1].set_title('Histogram of Average for Males', fontsize=16)

    sns.histplot(data=df[df.gender == 'female'], x='average', kde=True, hue='parental_education', ax=axs[2])
    axs[2].set_xlabel('Average', fontsize=14)
    axs[2].set_ylabel('Frequency', fontsize=14)
    axs[2].set_title('Histogram of Average for Females', fontsize=16)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot9_image = "data:image/png;base64,{}".format(graphic)
     
    #plot10

    plt.figure(figsize=(18, 8))

    fig, axs = plt.subplots(1, 3, figsize=(18, 8))

    sns.violinplot(y='math score', data=df, color='red', linewidth=3, ax=axs[0])
    axs[0].set_title('MATH SCORES')
    axs[0].set_ylabel('Math Score', fontsize=14)

    sns.violinplot(y='reading_score', data=df, color='green', linewidth=3, ax=axs[1])
    axs[1].set_title('READING SCORES')
    axs[1].set_ylabel('Reading Score', fontsize=14)

    sns.violinplot(y='writing_score', data=df, color='blue', linewidth=3, ax=axs[2])
    axs[2].set_title('WRITING SCORES')
    axs[2].set_ylabel('Writing Score', fontsize=14)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot10_image = "data:image/png;base64,{}".format(graphic)
    

    f, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Countplot
    sns.countplot(x=df['race_ethnicity'], data=df, palette='bright', ax=ax[0], saturation=0.95)
    for container in ax[0].containers:
        ax[0].bar_label(container, color='black', size=20)

    # Pie Chart
    ax[1].pie(
        x=df['race_ethnicity'].value_counts(),
        labels=df['race_ethnicity'].value_counts().index,
        explode=(0.1, 0, 0, 0, 0),
        autopct='%1.1f%%',
        shadow=True,
    )
    ax[1].set_title('Distribution of Race/Ethnicity')

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot11_image = "data:image/png;base64,{}".format(graphic)
 

    f, ax = plt.subplots(1, 2, figsize=(20, 8))


    sns.countplot(x=df['parental_education'], data=df, palette='bright', hue='test_prep_course', saturation=0.95, ax=ax[0])
    ax[0].set_title('Students vs Test Preparation Course', color='black', size=25)
    for container in ax[0].containers:
        ax[0].bar_label(container, color='black', size=20)

    
    sns.countplot(x=df['parental_education'], data=df, palette='bright', hue='lunch', saturation=0.95, ax=ax[1])
    for container in ax[1].containers:
        ax[1].bar_label(container, color='black', size=20)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png).decode('utf-8')
    plot13_image = "data:image/png;base64,{}".format(graphic)


    return render(request, 'index.html', {
            'model_results': model_results,
            'r2_scores': r2_scores,
            'form':form,
            'predictions':predictions,'plot1_image': plot1_image, 'plot2_image': plot2_image,
                   'plot3_image': plot3_image, 'plot4_image': plot4_image,
                   'plot5_image': plot5_image, 'plot6_image': plot6_image,
                   'plot7_image': plot7_image, 'plot8_image': plot8_image,
                   'plot9_image':plot9_image,'plot10_image':plot10_image,
                   'plot11_image':plot11_image,'plot13_image':plot13_image
                })