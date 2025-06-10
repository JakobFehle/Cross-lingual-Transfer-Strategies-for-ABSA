PROMPT_TEMPLATE = '''### Instruction:
{}

### Text:
{}

### Label:
{}'''

PROMPT_EN = '''According to the following sentiment elements definition: 
 
- The 'aspect category' refers to the category that aspect belongs to, and the available categories includes: [[aspect_category]].
- The 'sentiment polarity' refers to the degree of positivity, negativity or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, and the available polarities include: 'positive', 'negative' and 'neutral'.

Recognize all sentiment elements with their corresponding aspect categories and sentiment polarity in the following text with the format of [('aspect category', 'sentiment polarity'), ...].

[[examples]]'''

PROMPT_CS = '''Podle následující definice sentimentálních prvků:  
 
- 'Kategorie aspektu' označuje kategorii, do které aspekt patří, a dostupné kategorie zahrnují: [[aspect_category]].  
- 'Polarita sentimentu' označuje míru pozitivity, negativity nebo neutrality vyjádřenou v názoru na konkrétní aspekt nebo vlastnost produktu či služby. Dostupné polarity zahrnují: 'positive', 'negative' a 'neutral'.  

Rozpoznejte všechny sentimentální prvky s jejich odpovídajícími kategoriemi aspektů a polaritou sentimentu v následujícím textu ve formátu [('Kategorie aspektu', 'Polarita sentimentu'), ...].

[[examples]]'''

PROMPT_DE = '''Gemäß der folgenden Definition der Sentiment-Elemente:  

- Die 'Aspektkategorie' bezieht sich auf die Kategorie, zu der der Aspekt gehört, und die verfügbaren Kategorien sind: [[aspect_category]].  
- Die 'Sentiment-Polarität' beschreibt den Grad der Positivität, Negativität oder Neutralität, die in der Meinung zu einem bestimmten Aspekt oder Merkmal eines Produkts oder einer Dienstleistung ausgedrückt wird. Die verfügbaren Polaritäten sind: 'positive', 'negative' und 'neutral'.  

Erkenne alle Sentiment-Elemente mit ihren jeweiligen Aspektkategorien und Sentiment-Polaritäten im folgenden Text im Format [('Aspektkategorie', 'Sentiment-Polarität'), ...].

[[examples]]'''

PROMPT_ES = '''Según la siguiente definición de los elementos del sentimiento:  

- La 'categoría de aspecto' se refiere a la categoría a la que pertenece el aspecto, y las categorías disponibles incluyen: [[aspect_category]].  
- La 'polaridad del sentimiento' se refiere al grado de positividad, negatividad o neutralidad expresado en la opinión sobre un aspecto o característica particular de un producto o servicio. Las polaridades disponibles incluyen: 'positive', 'negative' y 'neutral'.  

Reconoce todos los elementos del sentimiento con sus categorías de aspecto y su polaridad del sentimiento en el siguiente texto con el formato [('categoría de aspecto', 'polaridad del sentimiento'), ...].

[[examples]]'''

PROMPT_FR = '''Selon la définition suivante des éléments de sentiment:  

- La 'catégorie d'aspect' fait référence à la catégorie à laquelle appartient l'aspect, et les catégories disponibles incluent : [[aspect_category]].  
- La 'polarité du sentiment' désigne le degré de positivité, de négativité ou de neutralité exprimé dans l'opinion à l'égard d'un aspect ou d'une caractéristique particulière d'un produit ou d'un service. Les polarités disponibles sont : 'positive', 'negative' et 'neutral'.  

Identifiez tous les éléments de sentiment avec leurs catégories d'aspect et leur polarité de sentiment dans le texte suivant, en utilisant le format [('catégorie d'aspect', 'polarité du sentiment'), ...].

[[examples]]'''

PROMPT_NL = '''Volgens de volgende definitie van sentimentselementen:  
 
- De 'aspectcategorie' verwijst naar de categorie waartoe het aspect behoort, en de beschikbare categorieën omvatten: [[aspect_category]].  
- De 'sentimentpolarisatie' verwijst naar de mate van positiviteit, negativiteit of neutraliteit die in de mening over een specifiek aspect of kenmerk van een product of dienst wordt uitgedrukt. De beschikbare polariteiten zijn: 'positive', 'negative' en 'neutral'.  

Identificeer alle sentimentselementen met hun bijbehorende aspectcategorieën en sentimentpolarisatie in de volgende tekst in het formaat [('aspectcategorie', 'sentimentpolarisatie'), ...].

[[examples]]'''

PROMPT_RU = '''Согласно следующему определению элементов сентимента:  

- 'Категория аспекта' – это категория, к которой относится аспект. Доступные категории включают: [[aspect_category]].  
- 'Полярность сентимента' обозначает степень положительности, отрицательности или нейтральности, выраженную в отношении конкретного аспекта или характеристики продукта или услуги. Доступные полярности: 'positive', 'negative' и 'neutral'.  

Распознайте все элементы сентимента с их соответствующими категориями аспектов и полярностью сентимента в следующем тексте в формате [('категория аспекта', 'полярность сентимента'), ...].

[[examples]]'''

PROMPT_TR = '''Aşağıdaki duygu öğeleri tanımına göre:  
 
- 'Özne kategorisi', öznenin ait olduğu kategoriyi ifade eder ve mevcut kategoriler şunlardır: [[aspect_category]].  
- 'Duygu kutuplaşması', bir ürün veya hizmetin belirli bir yönü veya özelliği hakkındaki görüşte ifade edilen olumlu, olumsuz veya tarafsız olma derecesini belirtir. Mevcut duygu kutuplaşmaları şunlardır: 'positive', 'negative' ve 'neutral'.  

Aşağıdaki metinde, ilgili özne kategorileri ve duygu kutuplaşmaları ile birlikte tüm duygu öğelerini şu formatta tanıyın: [('Özne kategorisi', 'Duygu kutuplaşması'), ...].

[[examples]]'''