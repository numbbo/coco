#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include <jni.h>

#include "coco.h"
#include "coco.c"
#include "javacoco_JNIinterface.h"

/*
 * Class:     javacoco_JNIinterface
 * Method:    coco_evaluate_function
 * Signature: (Ljavacoco/JNIinterface/Problem;[D)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_javacoco_JNIinterface_coco_evaluate_function
(JNIEnv * jenv, jobject obj, jobject problem, jdoubleArray x) {
    
    double *y; /* Result of evaluation. To be allocated with coco_allocate_vector(coco_get_number_of_objectives(pb)) */
    coco_problem_t *pb = NULL; /* Will contain the C problem */
    
    /* Necessary information to create the C problem */
    char *problem_suit;
    int function_index;
    char *observer;
    char *options;
    int nb_objectives;
    
    /* Java variables */
    jclass cls;
    jfieldID fid;
    jstring jproblem_suit;
    jint jfunction_index;
    jstring jobserver;
    jstring joptions;
    jdouble *cx;
    jdoubleArray jy; /* Returned double array */
    
    /* This test is both to prevent warning because obj was not used and check exceptions */
    if (obj == NULL)
        printf("Null obj found\n");
    
    /* Get attributes from jobj problem */
    cls = (*jenv)->GetObjectClass(jenv, problem);
    if (cls == NULL)
        printf("Null cls\n");
    
    /* Get problem_suit */
    fid = (*jenv)->GetFieldID(jenv, cls, "problem_suit", "Ljava/lang/String;");
    if(fid == NULL)
        printf("Null fid\n");
    jproblem_suit = (*jenv)->GetObjectField(jenv, problem, fid);
    problem_suit = (*jenv)->GetStringUTFChars(jenv, jproblem_suit, NULL);
    
    /* Get function_index */
    fid = (*jenv)->GetFieldID(jenv, cls, "function_index", "I");
    if(fid == NULL)
        printf("Null fid2\n");
    jfunction_index = (*jenv)->GetObjectField(jenv, problem, fid);
    function_index = (int)jfunction_index;
    
    /* Get observer */
    fid = (*jenv)->GetFieldID(jenv, cls, "observer_name", "Ljava/lang/String;");
    if(fid == NULL)
        printf("Null fid3\n");
    jobserver = (*jenv)->GetObjectField(jenv, problem, fid);
    observer = (*jenv)->GetStringUTFChars(jenv, jobserver, NULL);
    
    /* Get options */
    fid = (*jenv)->GetFieldID(jenv, cls, "options", "Ljava/lang/String;");
    if(fid == NULL)
        printf("Null fid4\n");
    joptions = (*jenv)->GetObjectField(jenv, problem, fid);
    options = (*jenv)->GetStringUTFChars(jenv, joptions, NULL);
    
    pb = coco_get_problem(problem_suit, function_index);
    pb = coco_observe_problem(observer, pb, options);
    
    /* Call coco_evaluate_function */
    cx = (*jenv)->GetDoubleArrayElements(jenv, x, NULL);
    nb_objectives = coco_get_number_of_objectives(pb);
    y = coco_allocate_vector(nb_objectives);
    coco_evaluate_function(pb, cx, y);
    
    /* Prepare the return value */
    jy = (*jenv)->NewDoubleArray(jenv, nb_objectives);
    (*jenv)->SetDoubleArrayRegion(jenv, jy, 0, nb_objectives, y);
    
    /* Free resources */
    coco_free_memory(y);
    coco_free_problem(pb);
    (*jenv)->ReleaseStringUTFChars(jenv, jproblem_suit, problem_suit);
    (*jenv)->ReleaseStringUTFChars(jenv, jobserver, observer);
    (*jenv)->ReleaseStringUTFChars(jenv, joptions, options);
    (*jenv)->ReleaseDoubleArrayElements(jenv, x, cx, JNI_ABORT);
    
    return jy;
}

