# Parser - Generación de AST

Este proyecto es un parser que genera un Árbol de Sintaxis Abstracta (AST) a partir de una serie de definiciones y expresiones en un lenguaje personalizado. El programa toma un archivo de entrada que contiene el código fuente y genera un archivo de salida con el AST correspondiente en formato JSON.
Requisitos

    Python 3.x
    Git Bash o cualquier terminal compatible para ejecutar el programa.

## Instrucciones de Uso
1. Clonar el Repositorio

Si aún no has descargado el proyecto, clona el repositorio usando el siguiente comando:

bash

git clone https://github.com/etolaba/parser.git

## 2. Ejecutar el Parser

Una vez que tengas el proyecto en tu máquina local, puedes ejecutar el parser desde la terminal usando el siguiente comando:

bash

python parser.py <caso_prueba.input> <salida.json>

Parámetros:

    <caso_prueba.input>: Es el archivo de texto que contiene el código fuente que deseas parsear.
    <salida.json>: Es el archivo donde se guardará el AST generado en formato JSON.

## 3. Ejemplo de Uso

Supongamos que tienes un archivo llamado codigo.txt que contiene el código fuente que deseas analizar. Para generar el AST y guardarlo en un archivo llamado resultado_ast.json, puedes ejecutar el siguiente comando:

python parser.py codigo.txt resultado_ast.json

Esto generará un archivo resultado_ast.json con la representación del AST del código fuente contenido en codigo.txt.

## 4. Salida

El archivo de salida será un archivo JSON formateado, con una estructura similar a la siguiente:

[
  ["Def", "nombre_definicion",
    ["ExprVar", "variable"]
  ],
  ["Def", "otra_definicion",
    ["ExprApply",
      ["ExprVar", "funcion"],
      ["ExprNumber", 123]
    ]
  ]
]