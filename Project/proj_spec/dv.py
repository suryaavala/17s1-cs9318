from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer


class RowIterator(TransformerMixin):
    """ Prepare dataframe for DictVectorizer """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (row for _, row in X.iterrows())


vectorizer = make_pipeline(RowIterator(), DictVectorizer())

# now you can use vectorizer as you might expect, e.g.
#vectorizer.fit_transform(df)
