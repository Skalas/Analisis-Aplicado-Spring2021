




' Ejemplo DCF Stack:

Sub DCF()

  Dim z As Double, CashFlow As Double, DiscRate As Double, Periods As Double: z = 0#

  CashFlow = InputBox("Enter initial cash flow: ", "Cash Flow")
  DiscRate = InputBox("Enter discount rate in decimal form: ", "Discount Rate")
  Periods = InputBox("How many periods (in years) are there?", "Periods")

  Dim i As Integer
  For i = 1 To Periods:
    z = z + CashFlow / (1# + DiscRate) ^ i
  Next

  MsgBox Format(z, "$0.00")

End Sub