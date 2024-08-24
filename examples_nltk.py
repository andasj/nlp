import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag, pos_tag_sents
import string
import matplotlib.pyplot as plt
from numpy.ma.core import angle

# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("tagsets")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")
# nltk.download('punkt_tab')
# nltk.download('tagsets_json')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')

Texto = (f"Nós somos feitos de poeira de estrelas. Nós somos uma maneira de o cosmos se autoconhecer. "
         f"A imaginação nos leva a mundos que nunca sequer existiram. Mas sem ela não vamos a lugar algum.  ")
print('\n')
print(Texto)

# nltk.download()

sentencas = sent_tokenize(Texto, language="portuguese")
print('\n')
print(type(sentencas))
print(len(sentencas))
print(sentencas)

tokens = word_tokenize(Texto, language="portuguese")
print('\n')
print(tokens)
print(len(tokens))

stops = stopwords.words("portuguese")
print('\n')
print(len(stops))
print(stops)

palavras_sem_stopwords = [p for p in tokens if p not in stops]
print('\n')
print(len(palavras_sem_stopwords))
print(Texto)
print(palavras_sem_stopwords)

print('\n')
print(string.punctuation)

print('\n')
palavras_sem_pontuacao = [p for p in palavras_sem_stopwords if p not in string.punctuation]
print(len(palavras_sem_pontuacao))
print(Texto)
print(palavras_sem_pontuacao)

frequencia = nltk.FreqDist(palavras_sem_pontuacao)
print('\n')
print(frequencia.most_common(20))
# frequencia.plot(30, cumulative=False)  # Exibe um gráfico com as 30 palavras mais frequentes
# plt.xticks(rotation=30)
# plt.show()

stemmer = PorterStemmer()
stem1  = [stemmer.stem(word) for word in palavras_sem_pontuacao]
print('\n')
print(palavras_sem_pontuacao)
print(stem1)

stemmer2 = SnowballStemmer("portuguese")
stem2 = [stemmer2.stem(word) for word in palavras_sem_pontuacao]
print('\n')
print(palavras_sem_pontuacao)
print(stem2)

stemmer3 = LancasterStemmer()
stem3 = [stemmer3.stem(word) for word in palavras_sem_pontuacao]
print('\n')
print(palavras_sem_pontuacao)
print(stem3)

# nltk.help.upenn_tagset()

pos = nltk.pos_tag(palavras_sem_pontuacao, lang="eng")
print('\n')
print(pos)

token2 = sent_tokenize(Texto)

ntokens = []
for tokensentenca in token2:
  ntokens.append(word_tokenize(tokensentenca))

print('\n')
print(ntokens)
possenteca = pos_tag_sents(ntokens)
print(possenteca)

lemmatizer = WordNetLemmatizer()
resultado = [lemmatizer.lemmatize(palavra) for palavra in palavras_sem_pontuacao]
print('\n')
print(palavras_sem_pontuacao)
print(resultado)

texto_en = "Barack Obama foi um presidente dos EUA"
token3 = word_tokenize(texto_en)
tags = pos_tag(token3)
en = nltk.ne_chunk(tags)
print('\n')
print(en)