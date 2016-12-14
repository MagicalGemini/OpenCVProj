package main

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lobjMatcher

#include <objMatcher.h>
#include <stdlib.h>
*/
import "C"
import (
	"bufio"
	"fmt"
	"os"
	"unsafe"
)

func main() {

	//imgFile, err := os.Open("test.jpg")
	imgFile, err := os.Open("object.png")
	if err != nil {
		fmt.Println("load img error: ", err.Error())
	}

	fileInfo, _ := imgFile.Stat()
	size := fileInfo.Size()
	bytes := make([]byte, size)
	buffer := bufio.NewReader(imgFile)
	_, err = buffer.Read(bytes)

	kps := C.detectKeyPoints((*C.char)(unsafe.Pointer(&bytes[0])), C.int(len(bytes)))

	serialData := C.GoString(kps)

	fmt.Println(serialData)
}
