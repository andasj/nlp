import spacy
from typing import Match
from spacy.matcher import Matcher
from spacy import displacy
nlp = spacy.load('pt_core_news_lg')

print(type(nlp))
print(nlp.pipe_names)

documento = nlp("As ações do Magazine Luiza S.A., Franca, Brasil, acumularam baixa de 70% ao ano. Assim já devolveram todos os ganhos do período da pandemeia")
print(len(documento.vocab))
print(type(documento))
print(documento)

for token in documento:
    print(token.text)

print("Tokens:           ", [token.text for token in documento])
print("Stop word:        ", [token.is_stop for token in documento])
print("Alfanumérico:     ", [token.is_alpha for token in documento])
print("Maiúsculo:        ", [token.is_upper for token in documento])
print("Pontuação:        ", [token.is_punct for token in documento])
print("Número:           ", [token.like_num for token in documento])
print("Sentença Inicial: ", [token.is_sent_start for token in documento])
print("Formato:          ", [token.shape_ for token in documento])

for token in documento:
    if token.like_num:
        print("Número encontrado: ", token.text)
    if token.is_punct:
        print("Pontuação encontrada; ", token.text)

for token in documento:
    print(token.text, " - ", token.pos_, " - ", token.dep_, " - ", token.lemma_, " - ", token.shape_)

for token in documento:
    print(token.text, " - ", token.morph)

for token in documento:
    print(token.text, " - ", token.tag_)

for ent in documento.ents:
    print(ent.text, " - ", ent.label_)

for token in documento:
    if token.is_stop:
        print("Stop word: ", token.text)

for words in nlp.Defaults.stop_words:
    print(words)

nlp.Defaults.stop_words.add("eita")
nlp.vocab['eita'].is_stop = True

token_lista = []
for token in documento:
    token_lista.append(token.text)

stop_lista = []
for words in nlp.Defaults.stop_words:
    stop_lista.append(words)

semstop = [word for word in token_lista if not word in stop_lista]

print(documento.text)
print(semstop)

print("Hash: ", nlp.vocab.strings["dados"])
print("Hash: ", documento.vocab.strings["dados"])
print("String: ", nlp.vocab.strings[6013848609874238634])

lex = nlp.vocab["dados"]
print(lex.text, " - ", lex.orth, " - ", lex.is_alpha, " - ", lex.is_lower)

#print(nlp("dados").vector.shape)
#print(nlp("dados").vector)
print(nlp("dados são uma nova forma de ver o mundo").vector)

documento1 = nlp("Ele viaja regularmente de carro")
documento2 = nlp("Ela regularmente de avião viaja")
print(documento1.similarity(documento2))
print(documento2.similarity(documento1))

documento3 = nlp("Devemos dizer comprimento ou cumprimento?")
tokenA = documento3[2]
print(tokenA)
tokenB = documento3[4]
print(tokenB)
print(tokenA.similarity(tokenB))

documento4 = nlp("Ele pede descrição. Ele pede discrição")
partA = documento4[0:3]
print(partA)
partB = documento4[4:7]
print(partB)
print(partA.similarity(partB))

documento5 = nlp("Você pode ligar para (51) - 9964656570 ou (11) 12344988 ")

matcher = Matcher(nlp.vocab)
padrao = [{"ORTH": "("}, {"SHAPE": "dd"}, {"ORTH": ")"}, {"ORTH": "-", "OP": "?"}, {"IS_DIGIT": True}]
matcher.add("telefone", [padrao])
matches = matcher(documento5)
for id, inicio, fim in matches:
    print(id)
    print(documento5[inicio:fim])

documento6 = nlp("Estamos infectados com micro organismos. MICROORGANISMOS são perigosos. Não enxergamos micro-organismos")
matcher = Matcher(nlp.vocab)
padrao1 = [{"LOWER": "micro-organismos"}]
padrao2 = [{"LOWER": "microorganismos"}]
padrao3 = [{"LOWER": "micro"}, {"LOWER": "organismos"}]

matcher.add("padrao", [padrao1, padrao2, padrao3])

matches = matcher(documento6)
for id, inicio, fim in matches:
    print(id)
    print(documento6[inicio:fim])

# Gerar a visualização de dependências
html = displacy.render(documento, style="dep", jupyter=False)
html1 = displacy.render(documento,style="dep", jupyter=False,
                options={'compact': True, 'distance': 120, 'color': '#FFFFFF', 'bg': '#000000', 'font': 'Arial'})

# Exibir o HTML (no PyCharm você pode salvar em um arquivo para visualizar)
with open("displacy_output.html", "w") as f:
    f.write(html)
with open("displacy_output1.html", "w") as f:
    f.write(html1)

print("Pipeline Normal: ", nlp.pipe_names)
nlp.remove_pipe('tok2vec')
print("Pipeline sem tok2vec: ", nlp.pipe_names)
nlp.add_pipe('tok2vec', after='morphologizer')
print("Pipeline Normal: ", nlp.pipe_names)
