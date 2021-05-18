Public Function ConeVolume(R As Double, h As Double) As Double
Dim pi As Double
pi = 4 * Atn(1)
ConeVolume = ((pi * (R ^ 2)) * h) / 3
End Function


Public Sub squareRootSb()
Dim x As Double, y As Double
x = InputBox("Enter a number")
y = Sqr(x)
ActiveCell = y
End Sub

Public Sub ejemploContar()
Dim nr, nc As Integer
nc = Selection.Columns.Count
nr = Selection.Rows.Count
MsgBox ("Your selection has " & nr & " rows and " & nc & " columns")
End Sub

Public Sub ejemploSuma1()
Dim x, y, z As Double
x = InputBox("Give me a number")
y = ActiveCell
z = x + y
' ponemos el resultado de la suma, una celda a la izquierda, una hacia abajo
ActiveCell.Offset(1, 1) = z
End Sub


Public Function Fluid(R As Double)
If R < 20000 Then
    Fluid = 16 / R
Else
    Fluid = 0.079 / R ^ (1 / 4)
End If
End Function


Public Function grade(num As Double) As String
If num >= 90 Then
    grade = "A"
ElseIf num >= 80 Then
    grade = "B"
ElseIf num >= 70 Then
    grade = "C"
ElseIf num >= 60 Then
    grade = "D"
Else
    grade = "F"
End If

End Function


Public Sub validateInput()
Dim p As Double
' Un tipo de while
Do
' Queremos que el usuario nos de un numero en (0,100)
    p = InputBox("Enter a percent conversion")
    If p <= 100 And p >= 0 Then Exit Do
    MsgBox ("Number must be in [0,100]")
Loop

End Sub

Public Function Divisible(n As Integer) As Integer
' Cuenta el numero de enteros hasta n divisibles entre 3 o 5
Dim i As Integer, c As Integer
c = 0
For i = 1 To n
If i Mod 3 = 0 Or i Mod 5 = 0 Then
    c = c + 1
End If
Next i
Divisible = c

End Function

Public Sub countFives()
' Importante
Dim i, nr, c As Integer
c = 0
' Oblenemos el tamaño del vector:
nr = Selection.Rows.Count
For i = 1 To nr
If Selection.Cells(i, 1) = 5 Then
    c = c + 1
End If
Next i
MsgBox ("There are " & c & " fives in your selection")

End Sub


' ARREGLOS 

Public Sub CreateArray()
' Declaramos un array de enteros
Dim A(2, 2) As Integer
A(1, 1) = 1
A(1, 2) = 4
A(2, 1) = 11
A(2, 2) = 2
Range("A1:B2") = A
End Sub