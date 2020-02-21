// print("hell0")

def showMsg(str: String, closure:Unit)={
    closure()
}

def printMsg()={print(str)}

showMsg("china", printMsg())

