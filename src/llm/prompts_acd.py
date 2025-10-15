PROMPT_TEMPLATE = '''### Instruction:
{}

### Text:
{}

### Label:
{}'''

PROMPT_EN = '''According to the following sentiment elements definition: 

- The 'aspect category' refers to the category that aspect belongs to, and the available categories includes: [[aspect_category]].

Recognize all sentiment elements with their corresponding aspect categories in the following text with the format of ['aspect category', ...].

[[examples]]'''

PROMPT_CS = '''Podle následující definice sentimentálních prvků:  

- 'Kategorie aspektu' označuje kategorii, do které aspekt patří, a dostupné kategorie zahrnují: [[aspect_category]].  

Rozpoznejte všechny sentimentální prvky s jejich odpovídajícími kategoriemi aspektů v následujícím textu ve formátu ['Kategorie aspektu', ...].

[[examples]]'''

PROMPT_DE = '''Gemäß der folgenden Definition der Sentiment-Elemente:  

- Die 'Aspektkategorie' bezieht sich auf die Kategorie, zu der der Aspekt gehört, und die verfügbaren Kategorien sind: [[aspect_category]].

Erkenne alle Sentiment-Elemente mit ihren jeweiligen Aspektkategorien im folgenden Text im Format ['Aspektkategorie', ...].

[[examples]]'''

PROMPT_ES = '''Según la siguiente definición de los elementos del sentimiento:  

- La 'categoría de aspecto' se refiere a la categoría a la que pertenece el aspecto, y las categorías disponibles incluyen: [[aspect_category]].

Reconoce todos los elementos del sentimiento con sus categorías de aspecto en el siguiente texto con el formato ['categoría de aspecto', ...].

[[examples]]'''

PROMPT_FR = '''Selon la définition suivante des éléments de sentiment:  

- La 'catégorie d'aspect' fait référence à la catégorie à laquelle appartient l'aspect, et les catégories disponibles incluent : [[aspect_category]].

Identifiez tous les éléments de sentiment avec leurs catégories d'aspect dans le texte suivant, en utilisant le format ['catégorie d'aspect', ...].

[[examples]]'''

PROMPT_NL = '''Volgens de volgende definitie van sentimentselementen:  

- De 'aspectcategorie' verwijst naar de categorie waartoe het aspect behoort, en de beschikbare categorieën omvatten: [[aspect_category]].

Identificeer alle sentimentselementen met hun bijbehorende aspectcategorieën in de volgende tekst in het formaat ['aspectcategorie', ...].

[[examples]]'''

PROMPT_RU = '''Согласно следующему определению элементов сентимента:  
 
- 'Категория аспекта' – это категория, к которой относится аспект. Доступные категории включают: [[aspect_category]].

Распознайте все элементы сентимента с их соответствующими категориями аспектов в следующем тексте в формате ['категория аспекта', ...].

[[examples]]'''

PROMPT_TR = '''Aşağıdaki duygu öğeleri tanımına göre:  

- 'Özne kategorisi', öznenin ait olduğu kategoriyi ifade eder ve mevcut kategoriler şunlardır: [[aspect_category]].  

Aşağıdaki metinde, ilgili özne kategorileri ile birlikte tüm duygu öğelerini şu formatta tanıyın: ['Özne kategorisi', ...].

[[examples]]'''