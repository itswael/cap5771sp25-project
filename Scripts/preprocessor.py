import numpy as np
import pandas as pd
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

    def handle_outliers(self):
        """
        Detects and removes rows with outliers using a majority voting approach.
        A value is considered an outlier in a numeric column if at least two out of three methods
        (IQR, Z-score, MAD) flag it.
        """
        # Identify numeric columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        outlier_indices = set()

        for col in numerical_cols:
            # Use non-null data for calculations
            col_data = self.df[col].dropna()

            # IQR Method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = set(col_data[(col_data < lower_bound) | (col_data > upper_bound)].index)

            # Z-score Method
            mean = col_data.mean()
            std = col_data.std()
            z_scores = (col_data - mean) / std

            # MAD Method
            median = np.median(col_data)
            abs_deviation = np.abs(col_data - median)
            mad = np.median(abs_deviation)
            mad_threshold = 3 * mad
            mad_outliers = set(col_data[np.abs(col_data - median) > mad_threshold].index)

            # Z-score Method
            z_outliers = set(col_data[np.abs(z_scores) > 3].index)

            # Democratic approach: flag index if flagged by at least 2 methods
            combined_indices = set.union(iqr_outliers, z_outliers, mad_outliers)
            for idx in combined_indices:
                count = 0
                if idx in iqr_outliers:
                    count += 1
                if idx in z_outliers:
                    count += 1
                if idx in mad_outliers:
                    count += 1
                if count >= 2:
                    outlier_indices.add(idx)

        outlier_columns = [col for col in numerical_cols if not self.df[col].dropna().index.isin(outlier_indices).all()]
        print(f"Removing {len(outlier_indices)} rows flagged as outliers across columns: {', '.join(outlier_columns)}.")
        self.df = self.df.drop(index=outlier_indices)
        
    def handle_duplicates(self):
        """
        Finds and drops duplicate rows. Before dropping duplicates, prints out one example
        of a duplicate record (with all of its duplicate occurrences).
        """
        duplicates = self.df[self.df.duplicated(keep=False)]
        if not duplicates.empty:
            # Pick the first duplicate row and print all occurrences of that record.
            first_dup_index = duplicates.index[0]
            duplicate_record = self.df.loc[first_dup_index]
            # Create a mask for all rows identical to the duplicate_record.
            mask = (self.df == duplicate_record).all(axis=1)
            dup_group = self.df[mask]
            print("Example duplicate group found:")
            print(dup_group)
        else:
            print("No duplicates found.")
            
        # Drop duplicate rows.
        self.df = self.df.drop_duplicates()