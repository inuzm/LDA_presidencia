# Análisis de temas en los discursos pronunciados por AMLO durante sus primeros 135 días de mandato vía *Latent Dirichlet Allocation*

Antes de comenzar quiero decir que comparto completamente la opinión de [este tuit de Omar](https://twitter.com/omar_pardo/status/1148998722761527296). Hoy en día es muy fácil usar modelos muy complicados en ambientes como [R](https://www.r-project.org/) porque las librerías que existen lo permiten, pero ¿qué tan bueno es usar algo que no entendemos? Digo, al final del día tenemos que interpretar las cosas y si no entendemos bien lo que usamos, difícilmente podremos interpretar las cosas. ¿no creen? Por lo tanto mi idea es que en este repositorio se explique un poco sobre *Latent Dirichlet Allocation*, un modelo muy popular en la modelación de temas, tanto que Julia Silge y David Robinson [escriben, mal, de éste en su libro](https://www.tidytextmining.com/topicmodeling.html) (sin explicar qué hay debajo del capó... mal mal mal) que ya ha sido mejorado y superado. Algunas alternativas las pueden consultar en [aquí](https://github.com/blei-lab).

## Un poco de los modelos para textos

Dos objetivos muy comunes cuando analizamos textos computacionalmente son el análisis de sentimiento y el modelado de temas. El primero busca clasificar como positivo y negativo un texto y el segundo busca encontrar los temas que subyacen una colección de textos. ¿Cómo es que la computadora puede hacer esto? Al final de cuentas los textos están compuestos de palabras y lo que obtenemos son resúmenes numéricos. ¡Hasta para las nubes de palabras! Por tanto necesitamos encontrar una forma de representar los textos de forma numérica. 

Un modelo muy común de representación es el de *bolsa de palabras*. La idea básica es considerar un diccionario de orden $K$ e identificar cada palabra con un vector en un espacio vectorial de dimensión $K$. Lo más usual es considerar $\mathbb{R}^K$ y considerar el mapeo que a cada palabra le asigne un vector canónico (uno cero cero cero cero..., cero uno cero cero cero cero..., etc). Bajo este mapeo es que podemos representar un texto como un vector de frecuencias, que no es más que una combinación lineal de los vectores canónicos muy intuitiva. Notemos que bajo esta representación para la computadora las siguientes oraciones serán indistinguibles:

    para nada, es un hombre bueno
    es un hombre bueno para nada

Primer chale. Esto significa que esta representación puede quitarle la connotación a algunas palabras. Una forma de evitar esto es usar [$n$-gramas](https://en.wikipedia.org/wiki/N-gram) aunque el incremento en complejidad rara vez significa una mejora sustancial en cuanto a los resultados (depende mucho de los datos). 

Sabiendo esto, qué hacemos si nuestro objetivo es obtener los temas principales de una colección de textos. ¿Hacemos una nube de palabras e inferimos de ésta como hacen [aquí](http://segasi.com.mx/post/100-d%C3%ADas-as%C3%AD-habl%C3%B3-l%C3%B3pez-obrador/)? ABSOLUTAMENTE NO. ¿Por qué? Las nubes de palabras en realidad no consideran la proporción de las palabras dentro de un texto, sino que juntan tooooooooodos los textos que estemos usando y nos da la frecuencia TOTAL de las palabras entre todos los textos. Aquí la polisemia nos puede jugar una mala pasada. Supongamos que tenemos los siguientes dos textos:

    yo cargo mi mochila todos los días
    el cargo que tiene es muy importante para la empresa

En la primera oración *cargo* es una acción mientras que en la segunda *cargo* tiene un significado completamente distinto. Si no tomamos en cuenta cómo se relaciona la palabra, dentro de cada uno de los textos, con otras y solamente nos fijamos en la frecuencia total entonces no estaremos entendiendo cabalmente de qué tema se trata cuando veamos la palabra *cargo*. Luego, inferir de nubes de palabras es ingenuo por decir lo menos. Segundo chale.

Así vamos viendo que, para modelar los temas, necesitaremos un modelo que considere las palabras dentro de cada uno de los textos que conformen nuestra colección. Además necesitaremos que los textos se relacionen entre sí de cierta manera, ¿si no para qué considerar más de un documento? Aquí es donde entra *Latent Dirichlet Allocation*.

## LDA

*Latent Dirichlet Allocation* es un modelo probabilístico de generación de textos introducido por Blei, Ng y Jordan en [este artículo del 2003](http://jmlr.org/papers/volume3/blei03a/blei03a.pdf). En este caso se supone que los textos son generados de la siguiente manera, suponiendo que hay una cantidad $\kappa < \infty$ de temas:
- La cantidad de palabras en el texto $i$, $N_i$, tendrá distribución Poisson con parámetro $\xi$.
- Dado $\alpha \in \mathbb{R}^{\kappa}$ con entradas no negativas (al menos una positiva), las proporciones de los temas en el texto $i$, denotado por un vector $\theta_i$, tendrá distribución Dirichlet con parámetro $\alpha$. Para los que no saben, la distribución Dirichlet es aquella que, cuando $\alpha$ es un vector con entradas positivas, tiene función de densidad para $(p_1, \ldots, p_{\kappa})$ (un vector con entradas positivas que sumen uno) dada por 
$$ 
\frac{\Gamma \left( \sum_{j = 1}^{\kappa} \alpha_j \right)}{\prod_{j = 1}^{\kappa} \Gamma(\alpha_j)} \prod_{j = 1}^{\kappa} p_j^{\alpha_j - 1}.
$$

- Condicional en $\theta_i$, el tema $\eta_{ij}$ (para $j \in \{1, \ldots, N_i\}$) tendrá distribución multinomial con parámetros $1$ y $\eta_{ij}$.
- Condicional en un tema $\eta_{ij}$ la palabra $w_{ij}$ (para $j \in \{1, \ldots, N_i\}$) tendrá distribución multinomial de tamaño $1$ y cuyo parámetro de probabilidad depende de $\eta_{ij}$.

Para entender un poco el último punto, supongamos que $\kappa = 3$ y $K = 2$, es decir que tenemos tres temas y un gran vocabulario de dos palabras. Podemos pensar que si toca el tema $1$ entonces las palabras tendrán distribución $\mathrm{Mult}(1, (0.4, 0.6))$, con el tema $2$ tendrá distribución $\mathrm{Mult}(1, (0.6, 0.4))$ y bajo el tema $3$ será la distribución $\mathrm{Mult}(1, (0.5, 0.5))$.

Eso es en realidad **LDA**. No es un método como describen Silge y Robinson sino un modelo. Ahora, ¿qué implica modelar los temas que subyazcan los textos por medio de LDA? Aquí sí es, en parte, lo que decriben Silge y Robinson. Por una parte se implica que cada texto es una mezcolanza de $\kappa$ temas y que cada tema será una mezcla ponderada de las $K$ palabras. ¡Pero aún hay más! Implícitamente estamos asumiendo que no nos importa el orden en el que aparezcan los textos (en el ámbito de temas creo que no es un supuesto taaaaaan descabellado) Y que dentro de los textos no nos interesa el orden en el que aparecen las palabras (primer chale).

¿Y qué onda con $\kappa$? Bueno, se pueden ajustar muchos modelos para varios valores y ver cuál da mejores resultados en cuanto a predicción o en función de la [perplejidad](https://en.wikipedia.org/wiki/Perplexity). O se puede ajustar de forma automática la cantidad de temas como se hace con un modelo no paramétrico como es [Hierarchical Dirichlet Process](https://people.eecs.berkeley.edu/~jordan/papers/hdp.pdf).

## Aplicación

Aprovechando que [Sebastián Garrido](https://twitter.com/segasi) hizo una recolección de los discursos de AMLO durante sus primeros 135 como presidente, [disponible aquí btw](https://github.com/segasi/analisis_discursos_amlo_135_dias/tree/master/04_datos_output), y que no hizo un análisis de temas, decidí usar la misma base y LDA con el fin de obtener los temas subyacentes de los discursos.

El código de R se encuentra en este mismo repositorio. Espero que sea intuitivo para los tidy-amantes porque como tidy-averso, uso lo menos posible las funciones. Por medio del paquete [tm](https://cran.r-project.org/web/packages/tm/tm.pdf) se quitaron palabras comunes, números y caracteres como `¡`, `¿`, etc. y se obtuvo una matriz documento-término en la cual una fila corresponde a un texto y una columna corresponde a una palabra de nuestro vocabulario.  En cada celda se encuentra la frecuencia de un término en un documento.

Así se pudo usar la función `LDA` del paquete [topicmodels](https://cran.r-project.org/web/packages/topicmodels/topicmodels.pdf) suponiendo que hay 20 temas en total. Se muestra la tabla de los temas y las 10 palabras más representativas de cada uno.

| Tema 1    | Tema 2     | Tema 3   | Tema 4  | Tema 5   | Tema 6   | Tema 7     | Tema 8  | Tema 9     | Tema 10   | Tema 11 | Tema 12  | Tema 13 | Tema 14     | Tema 15 | Tema 16  | Tema 17  | Tema 18    | Tema 19 | Tema 20      |
| --------- | ---------- | -------- | ------- | -------- | -------- | ---------- | ------- | ---------- | --------- | ------- | -------- | ------- | ----------- | ------- | -------- | -------- | ---------- | ------- | ------------ |
| vamos     | vamos      | mil      | vamos   | mil      | mil      | vamos      | vamos   | mil        | vamos     | vamos   | vamos    | vamos   | vamos       | vamos   | mil      | mil      | vamos      | vamos   | mil          |
| mil       | petróleo   | pueblo   | oaxaca  | vamos    | vamos    | país       | salud   | programa   | campeche  | mil     | méxico   | mil     | país        | mil     | vamos    | vamos    | van        | mil     | constitución |
| aquí      | méxico     | van      | pueblo  | pesos    | pesos    | pueblo     | van     | vamos      | país      | pesos   | país     | méxico  | inversión   | van     | van      | pesos    | nuevo      | pueblo  | corrupción   |
| van       | aquí       | vamos    | aquí    | van      | pueblo   | gobierno   | oaxaca  | gobierno   | seguridad | méxico  | mil      | aquí    | sector      | pueblo  | pesos    | pueblo   | mil        | van     | presidente   |
| pesos     | país       | pesos    | mil     | pueblo   | veracruz | méxico     | pueblos | aquí       | roo       | van     | gobierno | pueblo  | crecimiento | méxico  | pueblo   | aquí     | aquí       | pesos   | méxico       |
| pueblo    | corrupción | puebla   | méxico  | méxico   | van      | ejército   | mil     | pueblo     | quintana  | pueblo  | programa | van     | desarrollo  | pesos   | aquí     | jóvenes  | pesos      | jóvenes | millones     |
| méxico    | mil        | jóvenes  | ustedes | gobierno | dos      | corrupción | sistema | presidente | aquí      | jóvenes | pueblo   | pesos   | salario     | jóvenes | jóvenes  | méxico   | aeropuerto | apoyo   | pesos        |
| michoacán | por        | programa | por     | jóvenes  | jóvenes  | por        | aquí    | van        | pública   | dos     | sureste  | ahora   | méxico      | aquí    | méxico   | millones | tener      | dos     | año          |
| ahora     | mujeres    | gobierno | van     | gente    | méxico   | pública    | trabajo | pesos      | ahora     | aquí    | van      | dos     | gobierno    | ahora   | programa | programa | león       | méxico  | gobierno     |
| jóvenes   | petrolera  | dos      | país    | ahora    | apoyo    | poder      | años    | bienestar  | elementos | ahora   | pueblos  | jóvenes | mil         | ser     | dos      | van      | ahora      | ser     | aquí         |

A ojo de buen cubero parece que hay palabras que están sesgando un poco los resultados: `vamos`, `mil` y `van` por  ejemplo. La verdad es que en esta parte del análisis es donde termina mi trabajo y debe comenzar el de una peronsa cuya experiencia se incline más hacia asuntos relacionados con política, pues con mayor facilidad y certeza (o eso espero yo) determinará el tema correspondiente a cada grupo de palabras. Por mi parte surgen preguntas como
- ¿Qué tanta información nos da la palabra mil? 
- ¿Y la palabra México? 
- ¿Qué pasará si agrupamos las palabras con una raíz común, esto es lematizar las palabras? 
- ¿Qué palabras comunes hay en los discursos del presidente?

Este post fue patrocinado por la procrastinación.