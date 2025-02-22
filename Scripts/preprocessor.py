class Preprocessor:
    def __init__(self, df):
        """
        Initialize the preprocessor with a copy of the DataFrame.
        """
        self.df = df.copy()

    def handle_nulls(self, numeric_method='mean', categorical_method='mode', drop=False, columns=None):
        """
        Handles null values in the DataFrame.

        Parameters:
          - numeric_method: For numeric columns, choose one of 'mean', 'median', 'mode',
                            or 'drop' to drop rows with nulls in that column.
          - categorical_method: For string columns, choose 'mode' to fill with mode,
                                supply a default value, or 'drop' to drop rows with nulls.
          - drop: If True, drop rows with any null values. (Ignored if columns is provided.)
          - columns: Optional list of column names to handle null values for.
                     If not provided, all columns will be handled.
        """
        if columns is None:
            # Handle all columns
            if drop:
                self.df = self.df.dropna()
            else:
                # Handle numeric columns
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_cols:
                    self._handle_null_column(col, numeric_method)

                # Handle categorical columns
                categorical_cols = self.df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    self._handle_null_column(col, categorical_method)
        else:
            # Handle specified columns
            for col in columns:
                if col not in self.df.columns:
                    print(f"Column '{col}' does not exist in the DataFrame.")
                    continue
                self._handle_null_column(col, numeric_method if self.df[col].dtype in ['int64',
                                                                                       'float64'] else categorical_method)

    def _handle_null_column(self, col, method):
        """
        Handles null values for a single column.

        Parameters:
          - col: the column to handle null values for.
          - method: the method to use for handling null values.
        """
        if self.df[col].isnull().sum() > 0:
            if method == 'drop':
                self.df = self.df[self.df[col].notnull()]
            elif method == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif method == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif method == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                self.df[col].fillna(method, inplace=True)