# Ponemos el directorio donde se encuentra el script y los datos
# Obviamente no pongo mis directorios :P
setwd("~/Donde/estan/los/datos")

# Leemos los datos. Son los que están disponibles en 
# https://github.com/segasi/analisis_discursos_amlo_135_dias/tree/master/04_datos_output
discursos <- readxl::read_xls("./discursos_amlo.xls")

# Cargamos la paquetería tm porque muerte a tidyverse, además LDA funciona con matrices
# ralas de documentos y términos. Además cargamos el paquete topicmodels porque es en
# el que está implementado LDA de forma variacional con el código de Blei et al en C++.
require(tm)
require(topicmodels)

# Creamos el corpus de los discursos
discursos_corpus <- VCorpus(VectorSource(discursos$texto))
 
# Si construimos una matriz de documentos y términos veremos que hay muchos términos 
# comunes. Éstos en general no nos aportan información por lo que podemos quitarlos.

discursos_corpus_sin_pc <- tm_map(discursos_corpus, removeWords, stopwords("es"))
discursos_corpus_casi <- tm_map(
    discursos_corpus_sin_pc, 
    tm_reduce,
    list(removePunctuation, removeNumbers)
)

# Podríamos ya hacer el análisis pero vemos que hay caracteres como '¿', '¡'
# que si no quitamos nos va a dar más palabras de las que en realidad hay.
# Convertimos estos caracteres extraños en espacios, los quitamos y finalmente
# removemos cadenas de espacios en blanco.
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
discursos_limpios <- tm_map(discursos_corpus_casi, toSpace, '\\“')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\–')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\¡')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\¿')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\‘')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\´')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\…')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\’')
discursos_limpios <- tm_map(discursos_limpios, toSpace, '\\”')
discursos_limpios <- tm_map(discursos_limpios, stripWhitespace)

# Parecería que ya está todo listo, así que creamos nuestra matriz de documentos
# y términos. :D
matriz_discursos <- DocumentTermMatrix(discursos_limpios)

# A veeeer qué hay en la matriz
inspect(matriz_discursos)

# Quitamos todo lo que ya no nos sirve. La memoria es preciosa.
rm(list = c(
    "discursos", 
    "discursos_corpus", 
    "discursos_corpus_sin_pc", 
    "discursos_corpus_casi", 
    "discursos_limpios"
))

# LDA de 20 temas. Creamos un control para imprimir el avance y modificar la distribución
# inicial, así como correr el algoritmo varias veces y quedarnos con el que tenga la
# mayor verosimilitud porque en estadística nos dicen que es mejor. 
# ¯\_(ツ)_/¯

RNGkind("Mars") # Cambiamos el generador de números pseudoaleatorios
set.seed(95473) # Ponemos la semilla

lda_control <- list(
    verbose = 1,
    nstart = 15,
    alpha = 1/20,
    seed = rpois(n = 15, rgamma(n = 15, shape = 1, rate = 1/1000)) # YOLO
)

# Corremos LDA con el control anterior. ヽ(^o^)丿
lda_discursos <- LDA(x = matriz_discursos, k = 20, control = lda_control)

# ¿Y si imprimimos las diez palabras más representativas de cada tema?  (?_?)
get_terms(lda_discursos, 10)
# CHA-LE

# Bueno, exportemos los resultados para poder hacer una linda tabla en markdown.
write.table(
    get_terms(lda_discursos, 10), 
    "./temas-palabras.txt", 
    sep = " | ", 
    row.names = FALSE, 
    col.names = FALSE,
    quote = FALSE
)

# Hay algunos detalles que se pueden mejorar para obtener un modelo mucho mejor. 
# Por propiedades del algoritmo, las palabras que aparezcan con mayor frecuencia van
# a tender a dominar en los temas. ¿Qué tanta información nos da la palabra mil? 
# ¿Y la palabra México? ¿Qué pasará si agrupamos las palabras con una raíz común, 
# esto es lematizar las palabras? 

# MUERTE A TIDYVERSE (nocierto)

# LDA de 5 temas. Creamos un control para imprimir el avance y modificar la distribución
# inicial, así como correr el algoritmo varias veces y quedarnos con el que tenga la
# mayor verosimilitud porque en estadística nos dicen que es mejor. 
# ¯\_(ツ)_/¯

RNGkind("Mars") # Cambiamos el generador de números pseudoaleatorios
set.seed(95473) # Ponemos la semilla

lda_control <- list(
    verbose = 1,
    nstart = 15,
    alpha = 1e-3,
    seed = rpois(n = 15, rgamma(n = 15, shape = 1, rate = 1/1000)), # YOLO
    var = list(iter.max = 1e3, tol = 1e-8),
    em = list(iter.max = 1e4, tol = 1e-8)
)

# Corremos LDA con el control anterior. ヽ(^o^)丿
lda_discursos <- LDA(x = matriz_discursos, k = 5, control = lda_control)

# ¿Y si imprimimos las diez palabras más representativas de cada tema?  (?_?)
get_terms(lda_discursos, 20)

# Exportemos los resultados para poder hacer una linda tabla en markdown.
write.table(
    get_terms(lda_discursos, 20), 
    "./temas-palabras-5thm.txt", 
    sep = " | ", 
    row.names = FALSE, 
    col.names = FALSE,
    quote = FALSE
)
