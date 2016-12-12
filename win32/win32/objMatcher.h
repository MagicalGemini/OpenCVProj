#ifndef __OBJ_MATCHER_H__
#define __OBJ_MATCHER_H__


#ifdef WIN32
#define EXPORT_DLL __declspec(dllexport)
#else
#define EXPORT_DLL
#endif

#ifdef __cplusplus
extern "C" 
{
#endif

	EXPORT_DLL const char* detectKeyPoints(char* buffer, int bufferSize);

	EXPORT_DLL int objMatchWithSerialData(char* imgBuffer, int imgBufferSize, char* serialBuffer, int serialBufferSize);

#ifdef __cplusplus
}//__cplusplus
#endif

#endif//__OBJ_MATCHER_H__

