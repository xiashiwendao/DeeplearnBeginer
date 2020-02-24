// print("hell0")

def showMsg(str: String, closure:Unit)={
    closure()
}

def printMsg()={print(str)}

showMsg("china", printMsg())

val lst = List("a", "b")

def matchTest(param: List[String]): Unit={
    param match{
        case List("a", "b")=> println("match")
        case _ => print("No match")
    }
}

matchTest(lst)

