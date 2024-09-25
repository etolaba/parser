import re
import json
import sys

# Definición de los tokens (ajustar el orden para que las palabras clave tengan prioridad)
TOKEN_REGEX = [
    ('DEF', r'\bdef\b'),  # Palabra clave 'def' primero
    ('LET', r'\blet\b'),  # Priorizar 'let'
    ('IN', r'\bin\b'),    # Priorizar 'in'
    ('IF', r'\bif\b'),
    ('THEN', r'\bthen\b'),
    ('ELIF', r'\belif\b'),
    ('ELSE', r'\belse\b'),
    ('CASE', r'\bcase\b'),
    ('SEMICOLON', r';'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('LAMBDA', r'\\'),
    ('OR', r'\|\|'),
    ('PIPE', r'\|'),
    ('ARROW', r'->'),
    ('EQ', r'=='),
    ('NE', r'!='),
    ('GE', r'>='),
    ('LE', r'<='),
    ('GT', r'>'),
    ('LT', r'<'),
    ('PLUS', r'\+'),
    ('MINUS', r'-'),
    ('TIMES', r'\*'),
    ('DIV', r'/'),
    ('MOD', r'%'),
    ('AND', r'&&'),
    ('NOT', r'!'),
    ('DEFEQ', r'='),
    ('NUMBER', r'\b[0-9]+\b'),
    ('CHAR', r"'(\\.|[^\\'])'"),  # Expresión regular para manejar caracteres con y sin escape
    ('STRING', r'"(\\.|[^\\"])*"'),
    ('LOWERID', r'\b[a-z][_a-zA-Z0-9]*\b'),  # Ahora 'LOWERID' viene después de 'in' y 'let'
    ('UPPERID', r'\b[A-Z][_a-zA-Z0-9]*\b'),
    ('WHITESPACE', r'\s+'),
    ('COMMENT', r'--.*'),  # Captura comentarios
]


def remove_comments(code):
    """Elimina comentarios que comienzan con '--' y descarta todo lo que está después de '--' en una línea, excepto dentro de cadenas."""
    result = []
    lines = code.splitlines()

    for line in lines:
        # Si la línea empieza con '--', es un comentario completo y la descartamos
        if line.strip().startswith('--'):
            continue
        
        in_string = False
        i = len(line) - 1  # Empezar desde el final de la línea
        while i >= 0:
            if line[i] == '"' and (i == 0 or line[i - 1] != '\\'):  # Detectar apertura/cierre de cadena
                in_string = not in_string
            elif i > 1 and line[i-1:i+1] == '--' and not in_string:  # Detectar comentario fuera de cadena
                line = line[:i-1].rstrip()  # Eliminar todo lo que está después de '--'
                break
            i -= 1
        
        if line.strip():  # Si la línea no está vacía después de eliminar el comentario
            result.append(line.strip())

    return '\n'.join(result)

def tokenize(code):
    tokens = []
    pos = 0
    in_string = False  # Bandera para saber si estamos dentro de una cadena
    lines = code.splitlines()  # Dividimos el código en líneas
    for line in lines:
        # Insertar espacio explícito entre los números y palabras clave 'def' para separar correctamente
        line = re.sub(r'(\d)(def)', r'\1 def', line)
        while pos < len(line):
            match = None
            for token_type, regex in TOKEN_REGEX:
                pattern = re.compile(regex)
                match = pattern.match(line, pos)
                
                if match:
                    # Identificar si estamos dentro o fuera de una cadena
                    if token_type == 'STRING':
                        in_string = not in_string  # Cambiar el estado de la cadena
                    # Ignorar espacios en blanco
                    if token_type != 'WHITESPACE':
                        tokens.append((token_type, match.group(0)))
                    pos = match.end(0)
                    break
            
            if not match:
                # Si estamos dentro de una cadena, no arrojar un error, ya que estamos buscando su cierre
                if in_string:
                    pos += 1
                else:
                    error_context = line[max(pos-10, 0):min(pos+10, len(line))]
                    raise SyntaxError(f"Unexpected character at position {pos}. Context: '{error_context}'")
        pos = 0  # Reiniciar el índice para la siguiente línea
    return tokens


# Clases del AST
class ASTNode:
    def to_json(self):
        raise NotImplementedError()

class Program(ASTNode):
    def __init__(self, definitions):
        self.definitions = definitions

    def to_json(self):
        return [definition.to_json() for definition in self.definitions]

class Definition(ASTNode):
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

    def to_json(self):
        return ["Def", self.name, self.expr.to_json()]

class ExprVar(ASTNode):
    def __init__(self, var_name):
        self.var_name = var_name

    def to_json(self):
        return ["ExprVar", self.var_name]
    
class ExprConstructor(ASTNode):
    def __init__(self, constructor_name):
        self.constructor_name = constructor_name

    def to_json(self):
        return ["ExprConstructor", self.constructor_name]

class ExprNumber(ASTNode):
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return ["ExprNumber", self.value]

class ExprChar(ASTNode):
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return ["ExprChar", self.value]
    
class ExprString(ASTNode):  # Nueva clase para manejar cadenas
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return ["ExprString", self.value]

class ExprApply(ASTNode):
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

    def to_json(self):
        return ["ExprApply", self.func.to_json(), self.arg.to_json()]
    
class ExprCase(ASTNode):  # Nueva clase para if-then-else
    def __init__(self, condition, branches):
        self.condition = condition
        self.branches = branches

    def to_json(self):
        return ["ExprCase", self.condition.to_json(), [branch.to_json() for branch in self.branches]]

class ExprLet(ASTNode):
    def __init__(self, var_name, value_expr, in_expr):
        self.var_name = var_name
        self.value_expr = value_expr
        self.in_expr = in_expr

    def to_json(self):
        return ["ExprLet", self.var_name, self.value_expr.to_json(), self.in_expr.to_json()]

class ExprLambda(ASTNode):
    def __init__(self, param, body):
        self.param = param
        self.body = body

    def to_json(self):
        return ["ExprLambda", self.param, self.body.to_json()]

class CaseBranch(ASTNode):
    def __init__(self, pattern, variables, result):
        self.pattern = pattern
        self.variables = variables
        self.result = result

    def to_json(self):
        return ["CaseBranch", self.pattern, self.variables, self.result.to_json()]

def save_ast_to_file(ast, filename):
    # Convertir el AST a JSON con una indentación de 2 espacios
    ast_json = json.dumps(ast.to_json(), indent=2, separators=(',', ': '))
    
    # Guardar el JSON formateado en un archivo
    with open(filename, 'w') as file:
        file.write(ast_json)
    print(f"AST generado y guardado en {filename}")


# Parser para construir el AST
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens  
        self.position = 0

    def previous_token(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position - 2]
        else:
            return None
        
    def next_expr_token(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position + 1]
        else:
            return None

    def current_token(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        else:
            return None

    def next_token(self):
        if self.position < len(self.tokens):
            self.position += 1

    def consume(self, expected_type, in_let_context=False):
        token = self.current_token()

        # Si estamos en el contexto de una ExprLet, verificar explícitamente la palabra clave 'IN'
        if in_let_context and expected_type == 'IN' and token and token[0] == 'LOWERID' and token[1] == 'in':
            # Si encontramos 'in' como un LOWERID, lo tratamos como palabra clave IN
            self.next_token()
            return ('IN', 'in')
        
        if token and token[0] == expected_type:
            self.next_token()
            return token
        else:
            raise SyntaxError(f"Expected {expected_type}, but got {token[0] if token else 'EOF'}")

    
    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else None
    
    def rewind(self, steps=1):
        """Retrocede el índice de la posición actual de tokens."""
        self.position = max(self.position - steps, 0)

    def parse_program(self):
        definitions = []
        while self.peek() and self.peek()[0] == 'DEF':
            definitions.append(self.parse_definition())
        return Program(definitions)

    def parse_definition(self):
        self.consume('DEF')
        var_name = self.consume('LOWERID')[1]
        self.consume('DEFEQ')
        expr = self.parse_application()  # Aquí es donde ajustamos las aplicaciones
        return Definition(var_name, expr)

    def parse_application(self):
        """Parsea aplicaciones de funciones como `Cons 1 lista_vacia`."""
        left = self.parse_expr()
        while self.peek() and self.peek()[0] not in ('DEF', 'RPAREN', 'IN', 'LET', 'SEMICOLON'):
            right = self.parse_expr()
            left = ExprApply(left, right)
        return left

    def parse_expr(self):
        token = self.current_token()

        if token[0] == 'OR':  # Operador lógico OR
        # Retrocedemos para obtener el lado izquierdo del operador
            self.rewind()  # Retrocedemos para procesar el token anterior (aquí será 'a')
            left = self.parse_atom()  # Parseamos el operando izquierdo

            # Consumimos el operador OR
            self.consume('OR')

            # Obtenemos el lado derecho del operador
            right = self.parse_atom()  # Parseamos el operando derecho (después del OR)

            # Crear la aplicación del operador OR a los operandos
            return ExprApply(ExprApply(ExprVar("OR"), left), right)

        elif token[0] == 'AND':  # Operador lógico AND
            # Retrocedemos para obtener el lado izquierdo
            self.rewind()
            left = self.parse_atom()  # Parseamos el operando izquierdo

            # Consumimos el operador AND
            self.consume('AND')

            # Obtenemos el lado derecho
            right = self.parse_atom()  # Parseamos el operando derecho

            return ExprApply(ExprApply(ExprVar("AND"), left), right)

        elif token[0] == 'NOT':  # Operador lógico NOT (unario)
            self.consume('NOT')
            if self.peek()[0] in ('NUMBER', 'LOWERID', 'UPPERID', 'LPAREN', 'NOT'):
                expr = self.parse_atom()
                return ExprApply(ExprVar("NOT"), expr)
            else:
                raise SyntaxError("Expected an expression after NOT")

        elif token[0] == 'EQ':
            self.rewind()
            left = self.parse_atom()

            self.consume('EQ')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("EQ"), left), right)

        # Repetimos el mismo patrón para otros operadores binarios:
        elif token[0] == 'NE':
            self.rewind()
            left = self.parse_atom()

            self.consume('NE')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("NE"), left), right)

        elif token[0] == 'GE':
            self.rewind()
            left = self.parse_atom()

            self.consume('GE')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("GE"), left), right)

        elif token[0] == 'LE':
            self.rewind()
            left = self.parse_atom()

            self.consume('LE')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("LE"), left), right)

        elif token[0] == 'GT':
            self.rewind()
            left = self.parse_atom()

            self.consume('GT')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("GT"), left), right)

        elif token[0] == 'LT':
            self.rewind()
            left = self.parse_atom()

            self.consume('LT')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("LT"), left), right)

        elif token[0] == 'PLUS':
            self.rewind()
            left = self.parse_atom()

            self.consume('PLUS')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("ADD"), left), right)

        elif token[0] == 'MINUS':  # Operador MINUS puede ser unario o binario
            self.consume('MINUS')

            # Si el siguiente token es un número o una variable, es un operador unario
            if self.peek()[0] in ('NUMBER', 'LOWERID', 'UPPERID', 'LPAREN'):
                right = self.parse_atom()
                return ExprApply(ExprVar("UMINUS"), right)  # Operador unario negativo

            # Si hay un token a la izquierda, es un operador binario
            else:
                self.rewind()
                left = self.parse_atom()
                self.consume('MINUS')
                right = self.parse_atom()
                return ExprApply(ExprApply(ExprVar("SUB"), left), right)

        elif token[0] == 'TIMES':
            self.rewind()
            left = self.parse_atom()

            self.consume('TIMES')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("MUL"), left), right)

        elif token[0] == 'DIV':
            self.rewind()
            left = self.parse_atom()

            self.consume('DIV')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("DIV"), left), right)

        elif token[0] == 'MOD':
            self.rewind()
            left = self.parse_atom()

            self.consume('MOD')
            right = self.parse_atom()

            return ExprApply(ExprApply(ExprVar("MOD"), left), right)
        
        elif token[0] == 'IF':
            return self.parse_if_expr()
        elif token[0] == 'CASE':
            return self.parse_case_expr()
        elif token[0] == 'NUMBER':
            return self.parse_atom()
        elif token[0] == 'UPPERID':
            return self.parse_atom()
        elif token[0] == 'LOWERID':
            return self.parse_atom()
        elif token[0] == 'CHAR':
            return self.parse_char()
        elif token[0] == 'STRING':
            return self.parse_string()  # Aquí manejamos las cadenas
        elif token[0] == 'LPAREN':
            return self.parse_paren_expr()
        elif token[0] == 'LET': 
            return self.parse_let_expr()
        elif token[0] == 'LAMBDA':  # Nuevo manejo de funciones lambda
            return self.parse_lambda_expr()
        elif token[0] == 'SEMICOLON':  # Detectar secuenciación
            return self.parse_sequence(self.parse_expr())
        else:
            raise SyntaxError(f"Unexpected token in expression: {token}")
    
    def parse_if_expr(self):
        self.consume('IF')
        condition = self.parse_expr()
        self.consume('THEN')
        true_branch = self.parse_expr()
        branches = [CaseBranch("True", [], true_branch)]

        while self.peek() and self.peek()[0] in ('ELIF', 'ELSE'):
            if self.peek()[0] == 'ELIF':
                self.consume('ELIF')
                elif_condition = self.parse_expr()
                self.consume('THEN')
                elif_branch = self.parse_expr()
                branches.append(CaseBranch("False", [], ExprCase(elif_condition, [CaseBranch("True", [], elif_branch)])))
            elif self.peek()[0] == 'ELSE':
                self.consume('ELSE')
                else_branch = self.parse_expr()
                branches.append(CaseBranch("False", [], else_branch))
                break

        return ExprCase(condition, branches)

    def parse_case_expr(self):
        self.consume('CASE')
        condition = self.parse_expr()  # Parsear la expresión del caso
        branches = []

        while self.peek() and self.peek()[0] == 'PIPE':
            self.consume('PIPE')
            pattern = self.consume('UPPERID')[1]  # Obtener el constructor o patrón
            variables = []

            # Parsear variables del patrón, si hay
            while self.peek() and self.peek()[0] == 'LOWERID':
                variables.append(self.consume('LOWERID')[1])

            self.consume('ARROW')
            result = self.parse_expr()  # Parsear la expresión del resultado
            branches.append(CaseBranch(pattern, variables, result))

        return ExprCase(condition, branches)
    
    def parse_let_expr(self):
        """Parsea expresiones let-in, soportando múltiples parámetros en la declaración y permitiendo anidaciones."""
        self.consume('LET')
        var_name = self.consume('LOWERID')[1]  # Obtener el nombre de la variable o función

        # Verificar si hay parámetros adicionales para construir una ExprLambda anidada
        params = []
        while self.peek() and self.peek()[0] == 'LOWERID':
            params.append(self.consume('LOWERID')[1])

        self.consume('DEFEQ')  # Consumir el '='
        value_expr = self.parse_expr()  # Parsear la expresión del valor

        # Si hay parámetros, creamos ExprLambda anidados
        for param in reversed(params):
            value_expr = ExprLambda(param, value_expr)

        # Aquí verificamos explícitamente si tenemos un 'in', sin confundir con identificadores
        token = self.peek()
        if token and token[0] == 'IN':
            self.consume('IN')  # Consumimos 'in' solo si realmente es la palabra clave

        in_expr = self.parse_expr()  # Parsear la expresión dentro del 'in'

        # Si después del 'in' tenemos un LET, debemos procesar de forma recursiva.
        while self.peek() and self.peek()[0] == 'LET':
            in_expr = self.parse_let_expr()  # Llamada recursiva para manejar `let` anidados

        # Si es una aplicación múltiple (ejemplo: g f), debe ser manejada como una aplicación
        if self.peek() and self.peek()[0] not in ('DEF', 'RPAREN', 'IN'):
            in_expr = ExprApply(in_expr, self.parse_application())

        return ExprLet(var_name, value_expr, in_expr)

    def parse_lambda_expr(self):
        """Parsea una expresión lambda, permitiendo múltiples parámetros y cuerpos de lambdas anidados."""
        self.consume('LAMBDA')  # Consumimos el token `\`

        # Parseamos los parámetros de la función lambda
        params = []
        while self.peek() and self.peek()[0] == 'LOWERID':
            params.append(self.consume('LOWERID')[1])

        self.consume('ARROW')  # Consumimos la flecha `->`

        # Parseamos el cuerpo de la lambda
        body_expr = self.parse_expr()

        # Creamos lambdas anidadas si hay múltiples parámetros
        for param in reversed(params):
            body_expr = ExprLambda(param, body_expr)

        return body_expr


    def parse_char(self):
        token = self.consume('CHAR')
        char_value = token[1][1:-1]  # Remover comillas simples
        if len(char_value) == 1:
            return ExprChar(ord(char_value))  # Caracter normal
        elif char_value.startswith('\\'):  # Secuencia de escape
            escape_sequences = {
                't': 9,
                'n': 10,
                'r': 13,
                "'": 39,
                '"': 34,
                '\\': 92
            }
            return ExprChar(escape_sequences.get(char_value[1], ord(char_value[1])))
        else:
            return ExprChar(ord(char_value))
        
    def parse_string(self):
        token = self.consume('STRING')
        string_value = token[1][1:-1]  # Remover comillas dobles

        if string_value == "":
            # Verificar si el contexto es una aplicación de Cons
            if self.peek() and self.peek()[0] == 'UPPERID' and self.peek()[1] == 'Cons':
                return ExprConstructor("Nil")  # Tratar cadenas vacías como Nil en el contexto de Cons
            else:
                return ExprString("")  # Si no es en Cons, devolver como ExprString
        else:
            return ExprString(string_value)  # Para cadenas no vacías, ExprString normal

    def parse_application(self):
        """Parsea aplicaciones de funciones como `Cons 1 lista_vacia`."""
        left = self.parse_expr()
        while self.peek() and self.peek()[0] not in ('DEF', 'RPAREN', 'IN', 'LET', 'SEMICOLON'):
            right = self.parse_expr()
            left = ExprApply(left, right)
        return left

    def parse_sequence(self, left_expr):
        """
        Maneja secuencias de expresiones separadas por ';'.
        Anida las expresiones en una serie de ExprLet usando '_' como variable temporal.
        """
        while self.peek() and self.peek()[0] == 'SEMICOLON':
            self.consume('SEMICOLON')  # Consumir el ';'
            right_expr = self.parse_expr()  # Parsear la expresión de la derecha
            left_expr = ExprLet("_", left_expr, right_expr)  # Anidar las expresiones en ExprLet
        return left_expr

    def string_to_char_list(self, string_value):
        """Convierte una cadena en una secuencia de `Cons` con `ExprChar`."""
        expr = ExprConstructor("Nil")  # La lista vacía es Nil
        for char in reversed(string_value):
            expr = ExprApply(ExprApply(ExprConstructor("Cons"), ExprChar(ord(char))), expr)
        return expr

    def parse_atom(self):
        token = self.current_token()
        if token is None:
            raise SyntaxError("Unexpected end of input while parsing atom.")
        
        if token[0] == 'NUMBER':
            self.consume('NUMBER')
            return ExprNumber(int(token[1]))
        elif token[0] == 'UPPERID':
            self.consume('UPPERID')
            return ExprConstructor(token[1])
        elif token[0] == 'LOWERID':
            self.consume('LOWERID')
            return ExprVar(token[1])
        else:
            raise SyntaxError(f"Unexpected token: {token}")

    def parse_paren_expr(self):
        """Maneja expresiones entre paréntesis."""
        self.consume('LPAREN')
        expr = self.parse_application()  # Se parsea como una aplicación de funciones
        self.consume('RPAREN')
        return expr

# Ejecución
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python parser.py <archivo_entrada.txt> <archivo_salida_ast.txt>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as f:
        code = f.read()

    code_without_comments = remove_comments(code)  # Eliminar comentarios
    tokens = tokenize(code_without_comments)  # Tokenizar el código sin comentarios
    print("Tokens generados:", tokens)

    parser = Parser(tokens)
    ast = parser.parse_program()
    save_ast_to_file(ast, output_file)
