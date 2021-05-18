' Notas curso youtube


Sub SquareNumber()
Dim x As Double, y As Double
x = InputBox("Enter a number")
y = x^2
MsgBox("The square of your number is: " & y & ".")
End Sub

' El output es double
Function Savings(P as Double, i as Double, n as Integer) As Double 
Savings = P*((1+i)^n)
End Function



'Calcular la superficie de la casa
Public Function HouseSurfaceArea(HeightHouse As Double, WidthHouse As Double, DepthHouse As Double, HeightRoof As Double)
Dim Sides, FrontandBack, RoofSlant As Double 
Sides = HeightHouse * DepthHouse * 2
FrontandBack = ((HeightHouse * WidthHouse) + (HeightRoof * WidthHouse / 2)) * 2
RoofSlant = Sqr(HeightRoof * HeightRoof + WidthHouse / 2 * WidthHouse / 2)
' This is using the formula to calculate the length of a right triangle's slant.... A2 + B2 = C2
Roof = RoofSlant * DepthHouse * 2
HouseSurfaceArea = FrontandBack + Sides + Roof
End Function



Declarar una variable entera:
Dim aNumber as Integer (Dim: declare in memory)
Dim aDate as Date

' FOR LOOPS
Public Sub ForNext()
Dim x As Integer
For x = 1 To 10
' quiero que las primeras 10 entradas de la primera columna tengan el valor de 100
Cells(x, 1).Value = 100
' acaba el for
Next x
End Sub

' Doble for
Public Sub DoubleForNext()
Dim intCol As Integer
Dim intRow As Integer
' Hacemos 3 columnas de 10 renglones que diga wow
For intCol = 1 To 3
For intRow = 1 To 10
Cells(intRow, intCol).Value = "Wow!"
Next intRow
Next intCol
End Sub


'Triple For:
Public Sub TripleForNext()
Dim intCol As Integer
Dim intRow As Integer
Dim intSheet As Integer
For intSheet = 3 To 5
For intCol = 1 To 3
For intRow = 1 To 10

Worksheets(intSheet).Cells(intRow, intCol).Value = "Neat!"

Next intRow
Next intCol
Next intSheet

End Sub


'A una colección:
Public Sub ExForEach()
' aplicar un for a una coleccion
Dim x As Worksheet

For Each x In Worksheets
MsgBox "Found Worksheet: " & x.Name
Next x

End Sub


' Acabar un for si se cumple una condicion:
Public Sub ExitForExample()
Dim x As Integer
For x = 1 To 50
Range("B" & x).Select 'va cambiando el valor del renglon
If Range("B" & x).Value = "Stop" Then
    Exit For
ElseIf Range("B" & x).Value = "" Then
    Range("B" & x).Value = "Info"
End If
Next x
End Sub



'DO LOOPS
'Do While:
Public Sub ExampleDoWhileLoop()
Dim x As Integer
x = 1
Do While x < 10
Cells(x, 1).Value = 100 'recorremos la primera columna
x = x + 1
Loop 'acaba el loop
End Sub


Public Sub ExampleDoWhileCalcLoop()
Dim x As Integer
x = 5
' <> es lo mismo que !=
Do While Cells(x, 1).Value <> "" 
Cells(x, 3).Value = Cells(x, 2).Value + 30
x = x + 1
Loop
End Sub


Public Sub DoUntilLoopEx()
Dim intRow As Integer
intRow = 1
Do Until IsEmpty(Cells(intRow, 1))
Cells(intRow, 1).Value = "Info"
intRow = intRow + 1
Loop
End Sub



'Parent Macro:
Public Sub AllYourMacros()
' Call to all your macros at once using the "Call" statement:
Call FindYourData
Call CopyPasteinYourA4
Call InsYourHeaders
End Sub


Sub FindYourData()
'Declare a variable (String) to remember where the data starts
Dim datastart As String

' Start by selecting range A1
Range("A1").Select
Selection.End(xlDown).Select 'va hacia abajo hasta que cambie el patron
datastart = Selection.Address
Range(datastart).Select
End Sub


Sub CopyPasteinYourA4()
Selection.CurrentRegion.Select
Selection.Cut
Range("A4").Select
ActiveSheet.Paste
End Sub


Sub InsYourHeaders()

Range("A1").Select
Selection.Value = "Our Global Company"
Selection.Font.Bold = True
Selection.Font.Size = 16


Range("A3").Value = "Symbol"
Range("B3").Value = "Open"
Range("C3").Value = "Close"
Range("D3").Value = "Net Change"
Range("A3:D3").Font.Bold = True
Range("A3:D3").Font.Size = 12

Columns("A:D").AutoFit

End Sub



'INPUT BOXES

Public Sub Test()
inputreceived = MsgBox("Would you like to download the sunshine viruss?", vbYesNoCancel, "It's a beautiful day")
If inputreceived = 6 Then ' 6 es Yes
MsgBox "go play in the sunshine"
Else: MsgBox "stay inside"
End If
End Sub



'IF Then
Public Sub Testing123()

Dim strResponse As String
strResponse = InputBox("Which dwarf is your favourite?")
If strResponse = "Grumpy" Then
    MsgBox "I knew you were cool"
ElseIf strResponse = "Happy" Then
    MsgBox "What a positive outlook"
Else: MsgBox "wrong answer"
End If

End Sub



'SELECT

Public Sub Testing123()

Dim strResponse As String
strResponse = InputBox("Which dwarf is your favourite?")
Select Case strResponse 'metodo mas eficiente
    Case "Grumpy"
        MsgBox "cool"
    Case "Happy"
        MsgBox "nice"
    Case "Thor"
        MsgBox "wtf?"
    Case Else
        MsgBox "wrong answer"
End Select

End Sub


'Loco
Sub SortBy()

Dim Message, TitleBarTxt, DefaultTxt, SortVal As String ' declarar muchas variables al mismo tiempo
Dim YNAnswer As Integer 'Yes/No
Message = "Enter a number to sort by the following fields: " & vbCrLf & _
"1 - By Date and Time" & vbCrLf & _
"2 . By customer service rep, date and time"
' el (espacio)_ me permite seguir con el codigo en la sig linea

TitleBarTxt = "Sort Log"
DefaultTxt = "Enter 1 or 2"
SortVal = InputBox(Message, TitleBarTxt, DefaultTxt)

Select Case SortVal
    Case "1"
        Call DateThenTime
    Case "2"
        Call RepSort
    Case Else
        YNAnswer = MsgBox("You didn't type 1 or 2, try again?", vbYesNo)
        If YNAnswer = vbYes Then
            Call SortBy
        End If
End Select

End Sub

'Me quedé en min: 16 del curso



'UDF: User defined Functions

Public Function aCostForGas(MilesofStretch As Double, MilesPerGallon As Double, LocalGallonCost As Double)

aCostForGas = MilesofStretch / MilesPerGallon * LocalGallonCost

End Function


Public Function aPercentChange(Current As Double, Previous As Double)

aPercentChange = (Current - Previous) / Previous

End Function


Public Function aHouseSurfaceArea(HeightHouse As Double, WidthHouse As Double, DepthHouse As Double, HeightRoof As Double)

Dim Sides As Double ' declarar variables que vamos a usar
Dim FrontandBack As Double
Dim Roof As Double
Dim RoofSlant As Double
Sides = HeightHouse * DepthHouse * 2
FrontandBack = ((HeightHouse * WidthHouse) + (HeightRoof * WidthHouse / 2)) * 2
RoofSlant = Sqr(HeightRoof * HeightRoof + WidthHouse / 2 * WidthHouse / 2)
' This is using the formula to calculate the length of a right triangle's slant.... A2 + B2 = C2

Roof = RoofSlant * DepthHouse * 2
HouseSurfaceArea = FrontandBack + Sides + Roof

End Function



Range("A1").Value = "Our global company"
    Range("A2").Value = "Stock prices"
    ' Concatenate:
    Range("B4").Value = ActiveSheet.Name & " portfolio"
    


