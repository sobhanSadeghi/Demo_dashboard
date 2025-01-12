import streamlit as st
import streamlit as st
import pandas as pd
import sklearn 
from sklearn.preprocessing import StandardScaler,Normalizer,LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 
import plotly.express as px
import io
from datetime import datetime




# Session state initialization
if "data" not in st.session_state:
    st.session_state["data"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None 
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None


if 'log_table' not in st.session_state:
    st.session_state.log_table = pd.DataFrame(columns=["Timestamp", "Operation"])

def log_operation(operation, details=""):
    """
    Logs an operation performed on the dataset.

    :param operation: The type of operation performed.
    :param details: Optional details about the operation.
    """
    # Initialize the log text in session state if it doesn't exist
    if "log_text" not in st.session_state:
        st.session_state.log_text = ""

    # Append the new log entry to the existing log
    new_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {operation}: {details}\n"
    st.session_state.log_text += new_entry



def get_current_data():
    if st.session_state["processed_data"] is not None:
        return st.session_state["processed_data"]
    return st.session_state["data"]

def impute_by_mean(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' is not numeric and cannot be imputed with mean.")
    imp = SimpleImputer(strategy='mean')
    df[[column_name]] = imp.fit_transform(df[[column_name]])
    return df

def handle_missing_values(method, data, column=None, fill_value=None):
    if data is None:
        raise ValueError("No data provided.")
    if method == "drop":
        st.session_state["drop_na_data"] = data.dropna()
        log_operation("Handle Missing Values, ", "Method: Drop")
        return st.session_state["drop_na_data"]
    elif method == "fill":
        if column is None or fill_value is None:
            raise ValueError("Column and fill value must be specified for 'fill'.")
        if pd.api.types.is_integer_dtype(data[column]):
            fill_value = int(fill_value)  # Ensure fill value is an integer
        elif pd.api.types.is_float_dtype(data[column]):
            fill_value = float(fill_value)  # Ensure fill value is a float
        data[column] = data[column].fillna(fill_value)
        st.session_state["drop_na_data"] = data
        log_operation("Handle Missing Values, ", f"Method: Fill, Column: {column}, Fill Value: {fill_value}")
        return data
    elif method == "impute by mean":
        if column is None:
            raise ValueError("Column must be specified for 'impute'.")
        st.session_state["drop_na_data"] = impute_by_mean(data, column)
        log_operation("Handle Missing Values, ", f"Method: Impute Mean, Column: {column}")
        return st.session_state["drop_na_data"]
    else:
        raise ValueError(f"Invalid method '{method}'.")

def convert_column_type(data, column, new_type):
    try:
        if new_type == "date":
            data[column] = pd.to_datetime(data[column], errors="coerce").dt.date
        elif new_type=="date time":
                data[column] = pd.to_datetime(data[column], errors="coerce")
        elif new_type in ["int", "float"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
            if new_type == "int":
                data[column] = data[column].astype("Int64")
        else:
            data[column] = data[column].astype(new_type)
        st.session_state["processed_data"] = data
        log_operation("Convert Column Type, ", f"Column: {column}, New Type: {new_type}")
        return data
    except Exception as e:
        st.error(f"Error converting column '{column}' to '{new_type}': {e}")
        return data


def change_col_name(data,col,new_label):
    data = data.rename(columns={col:new_label})
    log_operation("Rename Column, ", f"From: {col}, To: {new_label}")
    return data

def drop_col(data,col):
    data=data.drop([col],axis=1)
    log_operation("Delete Column, ", f"Column: {col}")
    return data

def z_norm(data,col):
    scaler=StandardScaler()
    data[col]=scaler.fit_transform(data[[col]])
    log_operation("Normalize Column, ", f"Column: {col}, Method: Z-Score")
    return data

def norm(data,col):
    normalr=Normalizer()
    data[col]=normalr.fit_transform(data[[col]])
    log_operation("Normalize Column, ", f"Column: {col}, Method: Normalizer")
    return data

def duplicate(data):
    return data[data.duplicated()]

def drop_duplicate(data):
    data=data.drop_duplicates(ignore_index=True)
    return data

def label_encode(data,col):
    le = LabelEncoder()
    data[f"{col}_encoded"] = le.fit_transform(data[col])
    log_operation("Encode Column, ", f"Column: {col}, Method: Label Encoding")
    return data

def one_hot(data,col):
    ohe = OneHotEncoder(sparse_output=False)
    one_hot_encoded = ohe.fit_transform(data[[col]])
    one_hot_df = pd.DataFrame(
        one_hot_encoded,
        columns=[f"{col}_{category}" for category in ohe.categories_[0]],
        index=data.index
    )
    data = data.drop(columns=[col]).join(one_hot_df)
    log_operation("Encode Column, ", f"Column: {col}, Method: One-Hot Encoding")
    return data

def ordinal_encode(data,col):
    oe = OrdinalEncoder()
    data[col] = oe.fit_transform(data[[col]])
    return data

@st.dialog(" handel duplicated Rows")
def duplicated_dailog():
    data=get_current_data()
    duplicated_row=duplicate(data)
    if len(duplicated_row)>0:
        with st.expander("Rows that are duplicated"):
            st.dataframe(duplicated_row)
        st.write(f"Number of duplicated row: {len(duplicated_row)}")
        if st.button('drop duplicate '):
            st.session_state['processed_data']=drop_duplicate(data)
            st.success("Duplicates dropped successfully! Check updated data.")

    else:
        st.write('None duplicated Rows')    


@st.dialog(" Encode categorical columns")
def column_encoding():
    data=get_current_data()
    encode_column=st.selectbox("Choose a column to Encode",data.select_dtypes(include=['object']).columns)
    encode_type=st.selectbox("Choose Encoding type",['Label Encoding','One-Hot Encoding','Ordinal Encoding'])
    if st.button("Apply Encoding"):
        try:

            # Perform the selected encoding
            if encode_type == 'Label Encoding':
                st.session_state["processed_data"] = label_encode(data.copy(), encode_column)
                st.success(f"Label Encoding applied to column: {encode_column}")

            elif encode_type == 'One-Hot Encoding':
                st.session_state["processed_data"] = one_hot(data.copy(), encode_column)
                st.success(f"One-Hot Encoding applied to column: {encode_column}")

            elif encode_type == 'Ordinal Encoding':
                st.session_state["data"] = ordinal_encode(data.copy(), encode_column)
                st.success(f"Ordinal Encoding applied to column: {encode_column}")

          

        except Exception as e:
            st.error(f"Error while encoding column '{encode_column}': {e}")





@st.dialog("delete a column from data set")
def delete_dailog():
    data=get_current_data()
    name = st.selectbox("Choose a column to convert", data.columns)
    if st.button('delete a column'):
        try:
            st.session_state['processed_data']=drop_col(data.copy(),name)
            st.success(f"Column '{name}' successfully deleted .")
            st.rerun()  # Refresh the app to reflect changes
        except Exception as e:
            st.error(f"Error: {str(e)}")


@st.dialog("Handle Missing Values")
def handle_missing_dialog():
        data = get_current_data()
        method = st.selectbox("Choose a method to handle missing values", ["drop", "fill", "impute by mean", "impute by median"])
        selected_column = None
        fill_value = None

        if method in ["fill", "impute by mean", "impute by median"]:
            selected_column = st.selectbox("Select a column", data.columns)
            if method == "fill":
                fill_value = st.text_input("Enter a value to fill missing data")
        
        if st.button("Apply Missing Value Handling"):
            try:
                st.session_state["processed_data"] = handle_missing_values(
                    method, data.copy(), column=selected_column, fill_value=fill_value
                )
                st.success(f"Successfully handled missing values using '{method}' method.")
                st.rerun()  # Refresh the app to reflect changes
            except ValueError as e:
                st.error(f"Error: {str(e)}")


@st.dialog("Convert Data Types",width='large')
def convert_dialog():
    data = get_current_data()
    
    name = st.selectbox("Choose a column to convert", data.columns)
    new_type = st.selectbox("Choose a new data type", ["int", "float", "object", "date","date time"])
    
    if st.button("Convert Column"):
        try:
            st.session_state["processed_data"] = convert_column_type(data.copy(), name, new_type)
            st.success(f"Column '{name}' successfully converted to '{new_type}'.")
            st.rerun()  # Refresh the app to reflect changes
        except Exception as e:
            st.error(f"Error: {str(e)}")


@st.dialog("change column name",width="large")
def change_name_dailog():
    data = get_current_data()
    col_name=st.selectbox('choose a column',data.columns)
    new_name=st.text_input('new column name')
    
    if st.button('change name'):
        try:
            st.session_state['processed_data']=change_col_name(data.copy(),col_name,new_name)
            st.success(f"Column '{col_name}' successfully changed to '{new_name}'.")
            st.rerun()  # Refresh the app to reflect changes
        except Exception as e:
            st.error(f"Error: {str(e)}")

@st.dialog("regex")
def filter_regex():
    st.subheader("Filter Column with Regex")
    data=get_current_data()
    
    # Select a column to filter
    column_to_filter = st.selectbox("Select a column to filter", options=data.columns)
    
    # Input regex pattern
    regex_pattern = st.text_input("Enter the regex pattern to filter the column")
    
    if st.button("Apply Filter"):
        try:
            # Apply the regex filter
            filtered_data = data[data[column_to_filter].astype(str).str.contains(regex_pattern, na=False, regex=True)]
            
            if filtered_data.empty:
                st.warning("No rows matched the given regex pattern.")
            else:
                st.success(f"Filtered data contains {len(filtered_data)} rows.")
                st.dataframe(filtered_data, use_container_width=True)
            
            # Allow downloading the filtered data
            file_name = st.text_input("Name the filtered file (without extension)", value="filtered_data")
            if st.button("Download Filtered Data"):
                csv_data = filtered_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{file_name}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


@st.dialog("scale a numeric columnm values",width='large')
def scale_col():
    data=get_current_data()
    col_name=st.selectbox('choose a column',data.select_dtypes(include=['number']).columns)
    z_index=st.checkbox('use Z_index')
    normalize=st.checkbox('use normalization')
    

    if st.button('normalize'):

        if z_index:
            try:
                st.session_state['processed_data']=z_norm(data,col_name)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")    
        
        if normalize:
            try:
                st.session_state['processed_data']=norm(data,col_name)
                st.rerun()
            except Exception as e:
                st.error(f'Error:{str(e)}')    




@st.cache_data
def load_csv(file):
    return pd.read_csv(file)



# Define valid usernames and passwords
VALID_CREDENTIALS = {
    "admin": "password123",
    "user": "userpass",
    "guest": "guestpass",
    "browsAi":"123"
}

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "login"

# Function to handle login
def login_page():
    st.title("Login Page")

    # Input fields
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")

    # Login button
    if st.button("Login"):
        if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
            st.session_state.page = "welcome"  # Change page state
            st.rerun()  # Reload the app to reflect changes
        else:
            st.error("Invalid username or password. Please try again.")

# Function to handle welcome page
def welcome_page():
    
    st.title("Welcome to the Data Cleaning Application")

    # File uploader

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        # Check if the uploaded file is new
        if st.session_state["uploaded_file_name"] != uploaded_file.name:
            st.session_state["data"] =load_csv(uploaded_file)
            st.session_state["processed_data"] = None  # Reset processed data
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.toast("File uploaded and data loaded successfully!")
        else:
            pass
            



    # with st.sidebar:
    #     menu=st.radio(
    #     "Data Transform",
    #     ["Handle Missing Values", "Convert Data Types","change column name","delete column",'scale','Handle duplicate Rows','Encoding'],index=None)
    with st.sidebar:
        # List of options with their descriptions
        menu_options = {
            "Handle Missing Values": "Handle missing data by filling, dropping, or other imputation methods.",
            "Convert Data Types": "Convert columns to different data types (e.g., int to float).",
            "Change Column Name": "Rename a column in the dataset for better readability.",
            "Delete Column": "Remove unwanted or irrelevant columns from the dataset.",
            "Scale": "Scale numerical columns using standardization or normalization.",
            "Handle Duplicate Rows": "Identify and remove duplicate rows in the dataset.",
            "Encoding": "Encode categorical columns using techniques like one-hot or label encoding.",
            "Date formating": "change the formate of the date columns",
            "Filter Column with Regex" : "Filter rows based on a regex pattern applied to a column."

        }

        data = get_current_data()  
        if data is not None:
        # Dropdown for selecting one option
                selected_option = st.selectbox(
                    "**Data Transform options:**",
                    options=list(menu_options.keys()),
                    help="Choose one operation to perform on the data.",
                    index=None
                    
                )
                
                # Display the description for the selected option
                if selected_option:
                    st.markdown(f"**with this item:** {menu_options[selected_option]}")
                st.divider()
                st.write("Download Your Data")

                # Input for file name and file type
                file_name = st.text_input("Input a name for your file (without extension)", value="data")
                file_type = st.selectbox("Choose file type", ['csv', 'xlsx'])

            
                if not file_name.strip():
                    st.error("Please provide a valid file name.")
                else:
                    if file_type == "csv":
                        # Convert data to CSV
                        csv_data = data.to_csv(index=False,encoding='utf-8-sig')
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"{file_name}.csv",
                            mime="text/csv"
                        )
                    elif file_type == "xlsx":
                        # Convert data to Excel
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            data.to_excel(writer, index=False, sheet_name="Sheet1")
                            writer.save()
                        st.download_button(
                            label="Download Excel",
                            data=buffer,
                            file_name=f"{file_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )



    data = get_current_data()  


    if data is not None:
        # Create tabs
        tab1, tab2 = st.tabs(["Displaying the Data", "Data Summary"])
        
        # Tab 1: Display the data
        with tab1:
            st.title("Displaying the Data")
            st.dataframe(data,use_container_width=True)
            # Display the operation log
            st.title("Operation Log")
            if "log_text" not in st.session_state or not st.session_state.log_text.strip():
                st.text_area(" ", "No operations have been performed yet.", height=200, disabled=True)
            else:
                st.text_area(" ", st.session_state.log_text, height=200, disabled=True)




            
        
        # Tab 2: Data summary
        with tab2:
            st.title("Data Summary")
            # Initialize a list to collect statistics for each column
            stats_list = []
            
            # Loop through the columns and collect statistics
            for col_name in data.columns:
                col_stats = {
                    "Column": col_name,
                    "Data Type": str(data[col_name].dtypes),
                    "Count": len(data[col_name]),
                    "Distinct Values": len(data[col_name].unique()),
                    "Missing Values": data[col_name].isnull().sum()
                }
                stats_list.append(col_stats)
            
            # Convert the list of dictionaries to a DataFrame
            stats_df = pd.DataFrame(stats_list)
            
            # Display the DataFrame using Streamlit's column configuration
            st.dataframe(
                stats_df,
                use_container_width=True,
                column_config={
                    "Column": "Column Name",
                    "Data Type": "Data Type",
                    "Count": "Total Count",
                    "Distinct Values": "Unique Values",
                    "Missing Values": "Missing Values"
                }
            )
            st.header("Categorical and Numerical Column Analysis", divider=True)

            # Identify categorical and numerical columns
            string_col = data.select_dtypes(include=['object']).columns
            numeric_col = data.select_dtypes(include=['number']).columns

            # Analyze categorical columns
            for col_name in string_col:
                st.subheader(f"Categorical Column: {col_name}")
                
                # Create a row with two columns for details and plot
                col1, col2 = st.columns(2)

                with col1:
                    value_counts = data[col_name].value_counts()
                    top_values = " , ".join(value_counts.index[:3].astype(str))

                    # Display statistics
                    st.write(
                        f"**Statistics: Unique Values**: {len(data[col_name].unique())} ")

                    st.write( f"**Top 3 Values**: {top_values}")

                with col2:
                            # Get value counts
                    value_counts = data[col_name].value_counts()
                    
                    # Check the number of unique values
                    if len(value_counts) <= 4:
                        displayed_values = value_counts
                    else:
                        displayed_values = value_counts[:4]
                    
                    # Display progress bars for the selected values
                    st.write("Value Counts:")
                    total_count = len(data[col_name])
                    for value, count in displayed_values.items():
                        st.write(f"{value}: {count}")
                        st.progress(int(count / total_count * 100))

                st.divider()

            # Analyze numerical columns
            for col_name in numeric_col:
                st.subheader(f"Numerical Column: {col_name}")
                
                # Create a row with two columns for details and plot
                col1, col2 = st.columns(2)
                col_data = data[col_name]
                with col1:
                    

                    # Calculate statistics
                    mean = round(col_data.mean(), 2)
                    median = round(col_data.median(), 2)
                    std_dev = round(col_data.std(), 2)
                    min_val = col_data.min()
                    max_val = col_data.max()

                    # Detect outliers using the IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

                    # Count zero values
                    zero_values = (col_data == 0).sum()
                    zero_values_percentage = (zero_values / len(col_data)) * 100

                    # Display statistics
                    st.write(
                        f"Statistics: Mean: {mean} "
                    )
                    st.write(f"Median: {median}")
                    st.write(f"Std Dev: {std_dev}")
                    st.write(f"Outliers: {outliers_count}")
                    st.write(f"Zero Values: {zero_values}")
                    st.write(f"({zero_values_percentage:.2f}%)")

                with col2:
                    # Create a boxplot for value distribution
                    fig = px.box(
                        data,
                        y=col_name,
                        title=f"Boxplot for {col_name}",
                    
                    )
                    fig.update_layout(xaxis_title=f"{col_name}", yaxis_title="Values")
                    st.plotly_chart(fig, use_container_width=True, key=f"num_plot_{col_name}")

                st.divider()

        # Main processing
        if "data" in locals() and data is not None:  # Ensure 'data' exists and is not None
            if selected_option == "Handle Missing Values":
                handle_missing_dialog()

            elif selected_option == "Convert Data Types":
                convert_dialog()

            elif selected_option == "Change Column Name":
                change_name_dailog()

            elif selected_option == "Delete Column":
                delete_dailog()

            elif selected_option == "Scale":
                scale_col()

            elif selected_option == "Handle Duplicate Rows":
                duplicated_dailog()

            elif selected_option == "Encoding":
                column_encoding()

            elif selected_option=="Date formating":
                pass

            elif selected_option=="Filter Column with Regex":
                filter_regex()
        else:
            st.error("No data loaded. Please load data to proceed.")

    # Logout button
    if st.button("Logout"):
        st.session_state.page = "login"  # Change back to login page
        st.rerun()  # Reload the app

# Page routing
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "welcome":
    welcome_page()
