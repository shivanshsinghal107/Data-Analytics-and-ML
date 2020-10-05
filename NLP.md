## Tokenization [(Link)](https://www.youtube.com/watch?v=6ZVf1jnEKGI&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=2)
Tokenization means to convert the text/paragraph into words(`nltk.word_tokenize()`) or sentences(`nltk.sent_tokenize()`).

## Stemming & Lemmatization [(Link)](https://www.youtube.com/watch?v=JpxCt3kvbLk&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=3)
**Stop Words** - Most common words in a language which does not play much important role in predicting sentiment of a sentence(`nltk.corpus.stopwords()`).<br>
Process of reducing infected words to their word stem. Problem with this is that result given by stemming may not have any meaning(`nltk.stem.PorterStemmer()`).<br>
Lemmatization is a kind of stemming technique which is same as stemming with a difference that its result is always meaningful(`nltk.stem.WordNetLemmatizer()`).

## Bag of Words (BOW) [(Link)](https://www.youtube.com/watch?v=IKgBLTeQQL8&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=6)
A representation of text that describes the occurrence of words within a document/document matrix w.r.t the words. BOW is implemented using `CountVectorizer()`(`sklearn.feature_extraction.text.CountVectorizer()`).

## TF-IDF (Term Frequency-Inverse Document Frequency) [(Link)](https://www.youtube.com/watch?v=D2V1okCEsiE&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=8)
TF = No. of repetition of word in sentence/No. of words in sentence<br>
IDF = log(No. of sentences/No. of sentences containing words)

TF-IDF is implemented using `TfidfVectorizer()`(`sklearn.feature_extraction.text.TfidfVectorizer()`).

Problem with BOW and TFIDF is that both do not store the semantic information.

## Word2Vec [(Link)](https://www.youtube.com/watch?v=Otde6VGvhWM&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=11)
Every word is represented as a vector of 32 or more dimension instead of a single number. Here the semantic information and relation between different words is also preserved(`gensim.models.Word2Vec()`).

## Recurrent Neural Network(RNN) [(Link)](https://www.youtube.com/watch?v=CPl9XdIFbYA&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=15)
Most important feature due to which RNN is heavily used in NLP is storing sequence of information which is discarded in all other techniques Bag of Words, Tfidf or Word2Vec.

### Problem in RNN (Vanishing Gradient Problem & Exploding Gradient Problem) [(Link)](https://www.youtube.com/watch?v=mDaEfPgwtgo&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=18)
Steps taken during backpropagation i.e. the update of gradients keeps on becoming small(when using `sigmoid` as the activation function) after each iteration and eventually vanishes resulting in non-convergence to a global minimum. This problem is known as the **Vanishing Gradient Problem**.

Steps taken during backpropagation i.e. the update of gradients is so large(when using `relu` as the activation function) at each iteration which explodes the gradient/steps and do not let the weights converge to a global minimum. This problem is known as the **Exploding Gradient Problem**.

Now to tackle/overcome these problems, we use LSTM(Long Short Term Memory) which is a special kind of RNN.

## LSTM(Long Short Term Memory) [(Link)](https://www.youtube.com/watch?v=rdkIOM78ZPk&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=19) [(Reference)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- The main problem that LSTM solves is Long-Term Dependencies compare to simple RNN. Remembering information for long periods of time is practically their default behaviour.
- The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.
- Basically there are four components in LSTM:
  - Memory Cell
  - Forget Gate
  - I/p Layer
  - O/p Layer

### Steps-by-Step LSTM Walkthrough
- The first step in our LSTM is to decide what information we're going to throw away from the cell state. This decision is made by a sigmoid layer called the "forget gate layer".
- The next step is to decide what new information we're going to store in the cell state. This has two parts.
  - First, a sigmoid layer called the "input gate layer" decides which values we'll update.
  - Next, a tanh layer creates a vector of new candidate values that could be added to the state.
- In the next step, we'll combine these two to create an update to the state.

## Word Embedding
Word embeddings provide a dense representation of words and their relative meanings.
