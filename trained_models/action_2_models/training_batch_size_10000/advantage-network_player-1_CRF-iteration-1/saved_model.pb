??!
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
:
FloorMod
x"T
y"T
z"T"
Ttype:
	2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
|
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_27/kernel
u
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel* 
_output_shapes
:
??*
dtype0
s
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_27/bias
l
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes	
:?*
dtype0
|
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_28/kernel
u
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel* 
_output_shapes
:
??*
dtype0
s
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_28/bias
l
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes	
:?*
dtype0
{
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_30/kernel
t
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes
:	?*
dtype0
s
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_30/bias
l
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes	
:?*
dtype0
|
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_29/kernel
u
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel* 
_output_shapes
:
??*
dtype0
s
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_29/bias
l
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes	
:?*
dtype0
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
??*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:?*
dtype0
|
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_32/kernel
u
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel* 
_output_shapes
:
??*
dtype0
s
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_32/bias
l
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes	
:?*
dtype0
|
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_33/kernel
u
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel* 
_output_shapes
:
??*
dtype0
s
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_33/bias
l
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes	
:?*
dtype0
|
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_34/kernel
u
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel* 
_output_shapes
:
??*
dtype0
s
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_34/bias
l
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes	
:?*
dtype0
{
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_35/kernel
t
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes
:	?*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
embedding_20/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_20/embeddings
?
+embedding_20/embeddings/Read/ReadVariableOpReadVariableOpembedding_20/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_18/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_18/embeddings
?
+embedding_18/embeddings/Read/ReadVariableOpReadVariableOpembedding_18/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_19/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_19/embeddings
?
+embedding_19/embeddings/Read/ReadVariableOpReadVariableOpembedding_19/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_23/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*(
shared_nameembedding_23/embeddings
?
+embedding_23/embeddings/Read/ReadVariableOpReadVariableOpembedding_23/embeddings*
_output_shapes
:	4?*
dtype0
?
embedding_21/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_21/embeddings
?
+embedding_21/embeddings/Read/ReadVariableOpReadVariableOpembedding_21/embeddings*
_output_shapes
:	?*
dtype0
?
embedding_22/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameembedding_22/embeddings
?
+embedding_22/embeddings/Read/ReadVariableOpReadVariableOpembedding_22/embeddings*
_output_shapes
:	?*
dtype0
?
 normalize_3/normalization_3/meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" normalize_3/normalization_3/mean
?
4normalize_3/normalization_3/mean/Read/ReadVariableOpReadVariableOp normalize_3/normalization_3/mean*
_output_shapes	
:?*
dtype0
?
$normalize_3/normalization_3/varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$normalize_3/normalization_3/variance
?
8normalize_3/normalization_3/variance/Read/ReadVariableOpReadVariableOp$normalize_3/normalization_3/variance*
_output_shapes	
:?*
dtype0
?
!normalize_3/normalization_3/countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *2
shared_name#!normalize_3/normalization_3/count
?
5normalize_3/normalization_3/count/Read/ReadVariableOpReadVariableOp!normalize_3/normalization_3/count*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_27/kernel/m
?
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_27/bias/m
z
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_28/kernel/m
?
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_28/bias/m
z
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_30/kernel/m
?
*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_30/bias/m
z
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_29/kernel/m
?
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_29/bias/m
z
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/m
?
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/m
z
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_32/kernel/m
?
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_32/bias/m
z
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_33/kernel/m
?
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_33/bias/m
z
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_34/kernel/m
?
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_34/bias/m
z
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_35/kernel/m
?
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding_20/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_20/embeddings/m
?
2Adam/embedding_20/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_20/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_18/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_18/embeddings/m
?
2Adam/embedding_18/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_18/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_19/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_19/embeddings/m
?
2Adam/embedding_19/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_19/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_23/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_23/embeddings/m
?
2Adam/embedding_23/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_23/embeddings/m*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_21/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_21/embeddings/m
?
2Adam/embedding_21/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_21/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/embedding_22/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_22/embeddings/m
?
2Adam/embedding_22/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_22/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_27/kernel/v
?
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_27/bias/v
z
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_28/kernel/v
?
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_28/bias/v
z
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_30/kernel/v
?
*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_30/bias/v
z
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_29/kernel/v
?
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_29/bias/v
z
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/v
?
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_31/bias/v
z
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_32/kernel/v
?
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_32/bias/v
z
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_33/kernel/v
?
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_33/bias/v
z
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_34/kernel/v
?
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_34/bias/v
z
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_35/kernel/v
?
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:*
dtype0
?
Adam/embedding_20/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_20/embeddings/v
?
2Adam/embedding_20/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_20/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_18/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_18/embeddings/v
?
2Adam/embedding_18/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_18/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_19/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_19/embeddings/v
?
2Adam/embedding_19/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_19/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_23/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	4?*/
shared_name Adam/embedding_23/embeddings/v
?
2Adam/embedding_23/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_23/embeddings/v*
_output_shapes
:	4?*
dtype0
?
Adam/embedding_21/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_21/embeddings/v
?
2Adam/embedding_21/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_21/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/embedding_22/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name Adam/embedding_22/embeddings/v
?
2Adam/embedding_22/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_22/embeddings/v*
_output_shapes
:	?*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 * $tI
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *    

NoOpNoOp
??
Const_5Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
	variables
regularization_losses
trainable_variables
 	keras_api
!
signatures
 
 
 
?
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1	variables
2regularization_losses
3trainable_variables
4	keras_api
?
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api

H	keras_api

I	keras_api

J	keras_api

K	keras_api
h

Lkernel
Mbias
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api

R	keras_api
h

Skernel
Tbias
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
h

Ykernel
Zbias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
h

_kernel
`bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
h

ekernel
fbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api

k	keras_api
h

lkernel
mbias
n	variables
oregularization_losses
ptrainable_variables
q	keras_api

r	keras_api
h

skernel
tbias
u	variables
vregularization_losses
wtrainable_variables
x	keras_api

y	keras_api

z	keras_api
i

{kernel
|bias
}	variables
~regularization_losses
trainable_variables
?	keras_api

?	keras_api

?	keras_api
f
?	normalize
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26
 
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?
 ?layer_regularization_losses
?layers
	variables
?metrics
regularization_losses
?layer_metrics
?non_trainable_variables
trainable_variables
 
 
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2
 

?0
?1
?2
?
 ?layer_regularization_losses
?layers
1	variables
?metrics
2regularization_losses
?layer_metrics
?non_trainable_variables
3trainable_variables
 
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api
g
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?	keras_api

?0
?1
?2
 

?0
?1
?2
?
 ?layer_regularization_losses
?layers
D	variables
?metrics
Eregularization_losses
?layer_metrics
?non_trainable_variables
Ftrainable_variables
 
 
 
 
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
 

L0
M1
?
 ?layer_regularization_losses
?layers
N	variables
?metrics
Oregularization_losses
?layer_metrics
?non_trainable_variables
Ptrainable_variables
 
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
 

S0
T1
?
 ?layer_regularization_losses
?layers
U	variables
?metrics
Vregularization_losses
?layer_metrics
?non_trainable_variables
Wtrainable_variables
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
 

Y0
Z1
?
 ?layer_regularization_losses
?layers
[	variables
?metrics
\regularization_losses
?layer_metrics
?non_trainable_variables
]trainable_variables
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

_0
`1
 

_0
`1
?
 ?layer_regularization_losses
?layers
a	variables
?metrics
bregularization_losses
?layer_metrics
?non_trainable_variables
ctrainable_variables
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1
 

e0
f1
?
 ?layer_regularization_losses
?layers
g	variables
?metrics
hregularization_losses
?layer_metrics
?non_trainable_variables
itrainable_variables
 
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
?
 ?layer_regularization_losses
?layers
n	variables
?metrics
oregularization_losses
?layer_metrics
?non_trainable_variables
ptrainable_variables
 
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1
 

s0
t1
?
 ?layer_regularization_losses
?layers
u	variables
?metrics
vregularization_losses
?layer_metrics
?non_trainable_variables
wtrainable_variables
 
 
[Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

{0
|1
 

{0
|1
?
 ?layer_regularization_losses
?layers
}	variables
?metrics
~regularization_losses
?layer_metrics
?non_trainable_variables
trainable_variables
 
 
c
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api

?0
?1
?2
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
\Z
VARIABLE_VALUEdense_35/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_35/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_20/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_18/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_19/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_23/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_21/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEembedding_22/embeddings&variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE normalize_3/normalization_3/mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$normalize_3/normalization_3/variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!normalize_3/normalization_3/count'variables/24/.ATTRIBUTES/VARIABLE_VALUE
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25

?0
 

?0
?1
?2
 
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 
 
 
 
n
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
 
 
 
 
 
 
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 

?0
 

?0
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
 
 
 
 
 
 
n
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
&
	?mean
?variance

?count
 
 
 

?0
 
 

?0
?1
?2
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_35/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_35/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_20/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_18/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_19/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_23/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_21/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_22/embeddings/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_34/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_34/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_35/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_35/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_20/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_18/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_19/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_23/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_21/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/embedding_22/embeddings/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
w
serving_default_betsPlaceholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

y
serving_default_cards0Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_cards1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_betsserving_default_cards0serving_default_cards1ConstConst_1embedding_20/embeddingsembedding_18/embeddingsembedding_19/embeddingsConst_2embedding_23/embeddingsembedding_21/embeddingsembedding_22/embeddingsConst_3Const_4dense_27/kerneldense_27/biasdense_30/kerneldense_30/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/bias normalize_3/normalization_3/mean$normalize_3/normalization_3/variancedense_35/kerneldense_35/bias*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *0
f+R)
'__inference_signature_wrapper_400133562
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+embedding_20/embeddings/Read/ReadVariableOp+embedding_18/embeddings/Read/ReadVariableOp+embedding_19/embeddings/Read/ReadVariableOp+embedding_23/embeddings/Read/ReadVariableOp+embedding_21/embeddings/Read/ReadVariableOp+embedding_22/embeddings/Read/ReadVariableOp4normalize_3/normalization_3/mean/Read/ReadVariableOp8normalize_3/normalization_3/variance/Read/ReadVariableOp5normalize_3/normalization_3/count/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp2Adam/embedding_20/embeddings/m/Read/ReadVariableOp2Adam/embedding_18/embeddings/m/Read/ReadVariableOp2Adam/embedding_19/embeddings/m/Read/ReadVariableOp2Adam/embedding_23/embeddings/m/Read/ReadVariableOp2Adam/embedding_21/embeddings/m/Read/ReadVariableOp2Adam/embedding_22/embeddings/m/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOp2Adam/embedding_20/embeddings/v/Read/ReadVariableOp2Adam/embedding_18/embeddings/v/Read/ReadVariableOp2Adam/embedding_19/embeddings/v/Read/ReadVariableOp2Adam/embedding_23/embeddings/v/Read/ReadVariableOp2Adam/embedding_21/embeddings/v/Read/ReadVariableOp2Adam/embedding_22/embeddings/v/Read/ReadVariableOpConst_5*_
TinX
V2T		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__traced_save_400134860
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_30/kerneldense_30/biasdense_29/kerneldense_29/biasdense_31/kerneldense_31/biasdense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding_20/embeddingsembedding_18/embeddingsembedding_19/embeddingsembedding_23/embeddingsembedding_21/embeddingsembedding_22/embeddings normalize_3/normalization_3/mean$normalize_3/normalization_3/variance!normalize_3/normalization_3/counttotalcountAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/embedding_20/embeddings/mAdam/embedding_18/embeddings/mAdam/embedding_19/embeddings/mAdam/embedding_23/embeddings/mAdam/embedding_21/embeddings/mAdam/embedding_22/embeddings/mAdam/dense_27/kernel/vAdam/dense_27/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_30/kernel/vAdam/dense_30/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/vAdam/dense_31/kernel/vAdam/dense_31/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/vAdam/dense_33/kernel/vAdam/dense_33/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/vAdam/embedding_20/embeddings/vAdam/embedding_18/embeddings/vAdam/embedding_19/embeddings/vAdam/embedding_23/embeddings/vAdam/embedding_21/embeddings/vAdam/embedding_22/embeddings/v*^
TinW
U2S*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference__traced_restore_400135116??
?	
?
G__inference_dense_29_layer_call_and_return_conditional_losses_400132881

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_29_layer_call_fn_400134339

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_29_layer_call_and_return_conditional_losses_4001328812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
K__inference_embedding_21_layer_call_and_return_conditional_losses_400134560

inputs
embedding_lookup_400134554
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400134554Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400134554*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400134554*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
v
0__inference_embedding_23_layer_call_fn_400134550

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_23_layer_call_and_return_conditional_losses_4001325152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
K__inference_embedding_20_layer_call_and_return_conditional_losses_400132289

inputs
embedding_lookup_400132283
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400132283Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400132283*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400132283*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_28_layer_call_and_return_conditional_losses_400132854

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?Y
?

M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133160

cards0

cards1
bets+
'tf_math_greater_equal_11_greaterequal_y
model_6_400133075
model_6_400133077
model_6_400133079
model_6_400133081
model_7_400133084
model_7_400133086
model_7_400133088
model_7_400133090/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_400133102
dense_27_400133104
dense_30_400133107
dense_30_400133109
dense_28_400133112
dense_28_400133114
dense_29_400133117
dense_29_400133119
dense_31_400133122
dense_31_400133124
dense_32_400133129
dense_32_400133131
dense_33_400133135
dense_33_400133137
dense_34_400133142
dense_34_400133144
normalize_3_400133149
normalize_3_400133151
dense_35_400133154
dense_35_400133156
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualbets'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallcards0model_6_400133075model_6_400133077model_6_400133079model_6_400133081*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324662!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallcards1model_7_400133084model_7_400133086model_7_400133088model_7_400133090*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326922!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimumbets+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_400133102dense_27_400133104*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_27_layer_call_and_return_conditional_losses_4001328012"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_400133107dense_30_400133109*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_30_layer_call_and_return_conditional_losses_4001328272"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_400133112dense_28_400133114*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_28_layer_call_and_return_conditional_losses_4001328542"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_400133117dense_29_400133119*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_29_layer_call_and_return_conditional_losses_4001328812"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_400133122dense_31_400133124*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_31_layer_call_and_return_conditional_losses_4001329072"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_400133129dense_32_400133131*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_32_layer_call_and_return_conditional_losses_4001329352"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_400133135dense_33_400133137*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_33_layer_call_and_return_conditional_losses_4001329622"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_400133142dense_34_400133144*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_34_layer_call_and_return_conditional_losses_4001329902"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_400133149normalize_3_400133151*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_normalize_3_layer_call_and_return_conditional_losses_4001330252%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_400133154dense_35_400133156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_35_layer_call_and_return_conditional_losses_4001330512"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?9
?
F__inference_model_6_layer_call_and_return_conditional_losses_400134124

inputs*
&tf_math_greater_equal_9_greaterequal_y+
'embedding_20_embedding_lookup_400134098+
'embedding_18_embedding_lookup_400134104+
'embedding_19_embedding_lookup_400134112
identity??embedding_18/embedding_lookup?embedding_19/embedding_lookup?embedding_20/embedding_lookups
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_6/Const?
flatten_6/ReshapeReshapeinputsflatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_6/Reshape?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimumflatten_6/Reshape:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqualflatten_6/Reshape:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
embedding_20/CastCast$tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_20/Cast?
embedding_20/embedding_lookupResourceGather'embedding_20_embedding_lookup_400134098embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_20/embedding_lookup/400134098*,
_output_shapes
:??????????*
dtype02
embedding_20/embedding_lookup?
&embedding_20/embedding_lookup/IdentityIdentity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_20/embedding_lookup/400134098*,
_output_shapes
:??????????2(
&embedding_20/embedding_lookup/Identity?
(embedding_20/embedding_lookup/Identity_1Identity/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_20/embedding_lookup/Identity_1?
embedding_18/CastCast%tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_18/Cast?
embedding_18/embedding_lookupResourceGather'embedding_18_embedding_lookup_400134104embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_18/embedding_lookup/400134104*,
_output_shapes
:??????????*
dtype02
embedding_18/embedding_lookup?
&embedding_18/embedding_lookup/IdentityIdentity&embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_18/embedding_lookup/400134104*,
_output_shapes
:??????????2(
&embedding_18/embedding_lookup/Identity?
(embedding_18/embedding_lookup/Identity_1Identity/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_18/embedding_lookup/Identity_1?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV21embedding_20/embedding_lookup/Identity_1:output:01embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
embedding_19/CastCasttf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_19/Cast?
embedding_19/embedding_lookupResourceGather'embedding_19_embedding_lookup_400134112embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_19/embedding_lookup/400134112*,
_output_shapes
:??????????*
dtype02
embedding_19/embedding_lookup?
&embedding_19/embedding_lookup/IdentityIdentity&embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_19/embedding_lookup/400134112*,
_output_shapes
:??????????2(
&embedding_19/embedding_lookup/Identity?
(embedding_19/embedding_lookup/Identity_1Identity/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_19/embedding_lookup/Identity_1?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:01embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0^embedding_18/embedding_lookup^embedding_19/embedding_lookup^embedding_20/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_18/embedding_lookupembedding_18/embedding_lookup2>
embedding_19/embedding_lookupembedding_19/embedding_lookup2>
embedding_20/embedding_lookupembedding_20/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
G__inference_dense_27_layer_call_and_return_conditional_losses_400134271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_custom_model_3_layer_call_fn_400133483

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_3_layer_call_and_return_conditional_losses_4001334182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?9
?
F__inference_model_7_layer_call_and_return_conditional_losses_400134192

inputs+
'tf_math_greater_equal_10_greaterequal_y+
'embedding_23_embedding_lookup_400134166+
'embedding_21_embedding_lookup_400134172+
'embedding_22_embedding_lookup_400134180
identity??embedding_21/embedding_lookup?embedding_22/embedding_lookup?embedding_23/embedding_lookups
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_7/Const?
flatten_7/ReshapeReshapeinputsflatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_7/Reshape?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimumflatten_7/Reshape:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqualflatten_7/Reshape:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
embedding_23/CastCast%tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_23/Cast?
embedding_23/embedding_lookupResourceGather'embedding_23_embedding_lookup_400134166embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_23/embedding_lookup/400134166*,
_output_shapes
:??????????*
dtype02
embedding_23/embedding_lookup?
&embedding_23/embedding_lookup/IdentityIdentity&embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_23/embedding_lookup/400134166*,
_output_shapes
:??????????2(
&embedding_23/embedding_lookup/Identity?
(embedding_23/embedding_lookup/Identity_1Identity/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_23/embedding_lookup/Identity_1?
embedding_21/CastCast%tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_21/Cast?
embedding_21/embedding_lookupResourceGather'embedding_21_embedding_lookup_400134172embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_21/embedding_lookup/400134172*,
_output_shapes
:??????????*
dtype02
embedding_21/embedding_lookup?
&embedding_21/embedding_lookup/IdentityIdentity&embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_21/embedding_lookup/400134172*,
_output_shapes
:??????????2(
&embedding_21/embedding_lookup/Identity?
(embedding_21/embedding_lookup/Identity_1Identity/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_21/embedding_lookup/Identity_1?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV21embedding_23/embedding_lookup/Identity_1:output:01embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
embedding_22/CastCasttf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_22/Cast?
embedding_22/embedding_lookupResourceGather'embedding_22_embedding_lookup_400134180embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_22/embedding_lookup/400134180*,
_output_shapes
:??????????*
dtype02
embedding_22/embedding_lookup?
&embedding_22/embedding_lookup/IdentityIdentity&embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_22/embedding_lookup/400134180*,
_output_shapes
:??????????2(
&embedding_22/embedding_lookup/Identity?
(embedding_22/embedding_lookup/Identity_1Identity/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_22/embedding_lookup/Identity_1?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:01embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0^embedding_21/embedding_lookup^embedding_22/embedding_lookup^embedding_23/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_21/embedding_lookupembedding_21/embedding_lookup2>
embedding_22/embedding_lookupembedding_22/embedding_lookup2>
embedding_23/embedding_lookupembedding_23/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
K__inference_embedding_20_layer_call_and_return_conditional_losses_400134481

inputs
embedding_lookup_400134475
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400134475Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400134475*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400134475*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_32_layer_call_and_return_conditional_losses_400134368

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_7_layer_call_fn_400134247

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
G__inference_dense_33_layer_call_and_return_conditional_losses_400132962

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_7_layer_call_fn_400132703
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?
?
,__inference_dense_31_layer_call_fn_400134358

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_31_layer_call_and_return_conditional_losses_4001329072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_35_layer_call_and_return_conditional_losses_400134451

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_6_layer_call_fn_400134150

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
K__inference_embedding_21_layer_call_and_return_conditional_losses_400132537

inputs
embedding_lookup_400132531
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400132531Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400132531*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400132531*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_28_layer_call_and_return_conditional_losses_400134291

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_35_layer_call_fn_400134460

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_35_layer_call_and_return_conditional_losses_4001330512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_7_layer_call_and_return_conditional_losses_400134528

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133732

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_11_greaterequal_y2
.model_6_tf_math_greater_equal_9_greaterequal_y3
/model_6_embedding_20_embedding_lookup_4001335823
/model_6_embedding_18_embedding_lookup_4001335883
/model_6_embedding_19_embedding_lookup_4001335963
/model_7_tf_math_greater_equal_10_greaterequal_y3
/model_7_embedding_23_embedding_lookup_4001336203
/model_7_embedding_21_embedding_lookup_4001336263
/model_7_embedding_22_embedding_lookup_400133634/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource?
;normalize_3_normalization_3_reshape_readvariableop_resourceA
=normalize_3_normalization_3_reshape_1_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?%model_6/embedding_18/embedding_lookup?%model_6/embedding_19/embedding_lookup?%model_6/embedding_20/embedding_lookup?%model_7/embedding_21/embedding_lookup?%model_7/embedding_22/embedding_lookup?%model_7/embedding_23/embedding_lookup?2normalize_3/normalization_3/Reshape/ReadVariableOp?4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_6/flatten_6/Const?
model_6/flatten_6/ReshapeReshape
inputs_0_0 model_6/flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
model_6/flatten_6/Reshape?
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/y?
0model_6/tf.clip_by_value_9/clip_by_value/MinimumMinimum"model_6/flatten_6/Reshape:output:0;model_6/tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_6/tf.clip_by_value_9/clip_by_value/Minimum?
*model_6/tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_6/tf.clip_by_value_9/clip_by_value/y?
(model_6/tf.clip_by_value_9/clip_by_valueMaximum4model_6/tf.clip_by_value_9/clip_by_value/Minimum:z:03model_6/tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_6/tf.clip_by_value_9/clip_by_value?
+model_6/tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_6/tf.compat.v1.floor_div_6/FloorDiv/y?
)model_6/tf.compat.v1.floor_div_6/FloorDivFloorDiv,model_6/tf.clip_by_value_9/clip_by_value:z:04model_6/tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_6/tf.compat.v1.floor_div_6/FloorDiv?
,model_6/tf.math.greater_equal_9/GreaterEqualGreaterEqual"model_6/flatten_6/Reshape:output:0.model_6_tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_6/tf.math.greater_equal_9/GreaterEqual?
%model_6/tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_6/tf.math.floormod_6/FloorMod/y?
#model_6/tf.math.floormod_6/FloorModFloorMod,model_6/tf.clip_by_value_9/clip_by_value:z:0.model_6/tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_6/tf.math.floormod_6/FloorMod?
model_6/embedding_20/CastCast,model_6/tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_20/Cast?
%model_6/embedding_20/embedding_lookupResourceGather/model_6_embedding_20_embedding_lookup_400133582model_6/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_6/embedding_20/embedding_lookup/400133582*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_20/embedding_lookup?
.model_6/embedding_20/embedding_lookup/IdentityIdentity.model_6/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_6/embedding_20/embedding_lookup/400133582*,
_output_shapes
:??????????20
.model_6/embedding_20/embedding_lookup/Identity?
0model_6/embedding_20/embedding_lookup/Identity_1Identity7model_6/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_20/embedding_lookup/Identity_1?
model_6/embedding_18/CastCast-model_6/tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_18/Cast?
%model_6/embedding_18/embedding_lookupResourceGather/model_6_embedding_18_embedding_lookup_400133588model_6/embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_6/embedding_18/embedding_lookup/400133588*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_18/embedding_lookup?
.model_6/embedding_18/embedding_lookup/IdentityIdentity.model_6/embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_6/embedding_18/embedding_lookup/400133588*,
_output_shapes
:??????????20
.model_6/embedding_18/embedding_lookup/Identity?
0model_6/embedding_18/embedding_lookup/Identity_1Identity7model_6/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_18/embedding_lookup/Identity_1?
model_6/tf.cast_9/CastCast0model_6/tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_6/tf.cast_9/Cast?
%model_6/tf.__operators__.add_18/AddV2AddV29model_6/embedding_20/embedding_lookup/Identity_1:output:09model_6/embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_18/AddV2?
model_6/embedding_19/CastCast'model_6/tf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_19/Cast?
%model_6/embedding_19/embedding_lookupResourceGather/model_6_embedding_19_embedding_lookup_400133596model_6/embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_6/embedding_19/embedding_lookup/400133596*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_19/embedding_lookup?
.model_6/embedding_19/embedding_lookup/IdentityIdentity.model_6/embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_6/embedding_19/embedding_lookup/400133596*,
_output_shapes
:??????????20
.model_6/embedding_19/embedding_lookup/Identity?
0model_6/embedding_19/embedding_lookup/Identity_1Identity7model_6/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_19/embedding_lookup/Identity_1?
%model_6/tf.__operators__.add_19/AddV2AddV2)model_6/tf.__operators__.add_18/AddV2:z:09model_6/embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_19/AddV2?
'model_6/tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/tf.expand_dims_6/ExpandDims/dim?
#model_6/tf.expand_dims_6/ExpandDims
ExpandDimsmodel_6/tf.cast_9/Cast:y:00model_6/tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_6/tf.expand_dims_6/ExpandDims?
model_6/tf.math.multiply_6/MulMul)model_6/tf.__operators__.add_19/AddV2:z:0,model_6/tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_6/tf.math.multiply_6/Mul?
2model_6/tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_6/tf.math.reduce_sum_6/Sum/reduction_indices?
 model_6/tf.math.reduce_sum_6/SumSum"model_6/tf.math.multiply_6/Mul:z:0;model_6/tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_6/tf.math.reduce_sum_6/Sum?
model_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_7/flatten_7/Const?
model_7/flatten_7/ReshapeReshape
inputs_0_1 model_7/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
model_7/flatten_7/Reshape?
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/y?
1model_7/tf.clip_by_value_10/clip_by_value/MinimumMinimum"model_7/flatten_7/Reshape:output:0<model_7/tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_7/tf.clip_by_value_10/clip_by_value/Minimum?
+model_7/tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_7/tf.clip_by_value_10/clip_by_value/y?
)model_7/tf.clip_by_value_10/clip_by_valueMaximum5model_7/tf.clip_by_value_10/clip_by_value/Minimum:z:04model_7/tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.clip_by_value_10/clip_by_value?
+model_7/tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_7/tf.compat.v1.floor_div_7/FloorDiv/y?
)model_7/tf.compat.v1.floor_div_7/FloorDivFloorDiv-model_7/tf.clip_by_value_10/clip_by_value:z:04model_7/tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.compat.v1.floor_div_7/FloorDiv?
-model_7/tf.math.greater_equal_10/GreaterEqualGreaterEqual"model_7/flatten_7/Reshape:output:0/model_7_tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_7/tf.math.greater_equal_10/GreaterEqual?
%model_7/tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_7/tf.math.floormod_7/FloorMod/y?
#model_7/tf.math.floormod_7/FloorModFloorMod-model_7/tf.clip_by_value_10/clip_by_value:z:0.model_7/tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_7/tf.math.floormod_7/FloorMod?
model_7/embedding_23/CastCast-model_7/tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_23/Cast?
%model_7/embedding_23/embedding_lookupResourceGather/model_7_embedding_23_embedding_lookup_400133620model_7/embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_7/embedding_23/embedding_lookup/400133620*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_23/embedding_lookup?
.model_7/embedding_23/embedding_lookup/IdentityIdentity.model_7/embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_7/embedding_23/embedding_lookup/400133620*,
_output_shapes
:??????????20
.model_7/embedding_23/embedding_lookup/Identity?
0model_7/embedding_23/embedding_lookup/Identity_1Identity7model_7/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_23/embedding_lookup/Identity_1?
model_7/embedding_21/CastCast-model_7/tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_21/Cast?
%model_7/embedding_21/embedding_lookupResourceGather/model_7_embedding_21_embedding_lookup_400133626model_7/embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_7/embedding_21/embedding_lookup/400133626*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_21/embedding_lookup?
.model_7/embedding_21/embedding_lookup/IdentityIdentity.model_7/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_7/embedding_21/embedding_lookup/400133626*,
_output_shapes
:??????????20
.model_7/embedding_21/embedding_lookup/Identity?
0model_7/embedding_21/embedding_lookup/Identity_1Identity7model_7/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_21/embedding_lookup/Identity_1?
model_7/tf.cast_10/CastCast1model_7/tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_7/tf.cast_10/Cast?
%model_7/tf.__operators__.add_20/AddV2AddV29model_7/embedding_23/embedding_lookup/Identity_1:output:09model_7/embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_20/AddV2?
model_7/embedding_22/CastCast'model_7/tf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_22/Cast?
%model_7/embedding_22/embedding_lookupResourceGather/model_7_embedding_22_embedding_lookup_400133634model_7/embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_7/embedding_22/embedding_lookup/400133634*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_22/embedding_lookup?
.model_7/embedding_22/embedding_lookup/IdentityIdentity.model_7/embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_7/embedding_22/embedding_lookup/400133634*,
_output_shapes
:??????????20
.model_7/embedding_22/embedding_lookup/Identity?
0model_7/embedding_22/embedding_lookup/Identity_1Identity7model_7/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_22/embedding_lookup/Identity_1?
%model_7/tf.__operators__.add_21/AddV2AddV2)model_7/tf.__operators__.add_20/AddV2:z:09model_7/embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_21/AddV2?
'model_7/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_7/tf.expand_dims_7/ExpandDims/dim?
#model_7/tf.expand_dims_7/ExpandDims
ExpandDimsmodel_7/tf.cast_10/Cast:y:00model_7/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_7/tf.expand_dims_7/ExpandDims?
model_7/tf.math.multiply_7/MulMul)model_7/tf.__operators__.add_21/AddV2:z:0,model_7/tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_7/tf.math.multiply_7/Mul?
2model_7/tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_7/tf.math.reduce_sum_7/Sum/reduction_indices?
 model_7/tf.math.reduce_sum_7/SumSum"model_7/tf.math.multiply_7/Mul:z:0;model_7/tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_7/tf.math.reduce_sum_7/Sum?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2)model_6/tf.math.reduce_sum_6/Sum:output:0)model_7/tf.math.reduce_sum_7/Sum:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMultf.concat_9/concat:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_27/Relu?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMultf.concat_10/concat:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/BiasAdd?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/MatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_28/Relu?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_29/MatMul/ReadVariableOp?
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/MatMul?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/BiasAddt
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_29/Relu?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_30/BiasAdd:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2dense_29/Relu:activations:0dense_31/BiasAdd:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMultf.concat_11/concat:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd|
tf.nn.relu_9/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMultf.nn.relu_9/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAdd?
tf.__operators__.add_22/AddV2AddV2dense_33/BiasAdd:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul tf.nn.relu_10/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/BiasAdd?
tf.__operators__.add_23/AddV2AddV2dense_34/BiasAdd:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
2normalize_3/normalization_3/Reshape/ReadVariableOpReadVariableOp;normalize_3_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_3/normalization_3/Reshape/ReadVariableOp?
)normalize_3/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_3/normalization_3/Reshape/shape?
#normalize_3/normalization_3/ReshapeReshape:normalize_3/normalization_3/Reshape/ReadVariableOp:value:02normalize_3/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Reshape?
4normalize_3/normalization_3/Reshape_1/ReadVariableOpReadVariableOp=normalize_3_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
+normalize_3/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_3/normalization_3/Reshape_1/shape?
%normalize_3/normalization_3/Reshape_1Reshape<normalize_3/normalization_3/Reshape_1/ReadVariableOp:value:04normalize_3/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_3/normalization_3/Reshape_1?
normalize_3/normalization_3/subSub tf.nn.relu_11/Relu:activations:0,normalize_3/normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_3/normalization_3/sub?
 normalize_3/normalization_3/SqrtSqrt.normalize_3/normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_3/normalization_3/Sqrt?
%normalize_3/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_3/normalization_3/Maximum/y?
#normalize_3/normalization_3/MaximumMaximum$normalize_3/normalization_3/Sqrt:y:0.normalize_3/normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Maximum?
#normalize_3/normalization_3/truedivRealDiv#normalize_3/normalization_3/sub:z:0'normalize_3/normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_3/normalization_3/truediv?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMul'normalize_3/normalization_3/truediv:z:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdd?
IdentityIdentitydense_35/BiasAdd:output:0 ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp&^model_6/embedding_18/embedding_lookup&^model_6/embedding_19/embedding_lookup&^model_6/embedding_20/embedding_lookup&^model_7/embedding_21/embedding_lookup&^model_7/embedding_22/embedding_lookup&^model_7/embedding_23/embedding_lookup3^normalize_3/normalization_3/Reshape/ReadVariableOp5^normalize_3/normalization_3/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2N
%model_6/embedding_18/embedding_lookup%model_6/embedding_18/embedding_lookup2N
%model_6/embedding_19/embedding_lookup%model_6/embedding_19/embedding_lookup2N
%model_6/embedding_20/embedding_lookup%model_6/embedding_20/embedding_lookup2N
%model_7/embedding_21/embedding_lookup%model_7/embedding_21/embedding_lookup2N
%model_7/embedding_22/embedding_lookup%model_7/embedding_22/embedding_lookup2N
%model_7/embedding_23/embedding_lookup%model_7/embedding_23/embedding_lookup2h
2normalize_3/normalization_3/Reshape/ReadVariableOp2normalize_3/normalization_3/Reshape/ReadVariableOp2l
4normalize_3/normalization_3/Reshape_1/ReadVariableOp4normalize_3/normalization_3/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_flatten_6_layer_call_and_return_conditional_losses_400132261

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_signature_wrapper_400133562
bets

cards0

cards1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *-
f(R&
$__inference__wrapped_model_4001322512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????
:?????????:?????????: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:M I
'
_output_shapes
:?????????


_user_specified_namebets:OK
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?Y
?

M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133068

cards0

cards1
bets+
'tf_math_greater_equal_11_greaterequal_y
model_6_400132737
model_6_400132739
model_6_400132741
model_6_400132743
model_7_400132772
model_7_400132774
model_7_400132776
model_7_400132778/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_400132812
dense_27_400132814
dense_30_400132838
dense_30_400132840
dense_28_400132865
dense_28_400132867
dense_29_400132892
dense_29_400132894
dense_31_400132918
dense_31_400132920
dense_32_400132946
dense_32_400132948
dense_33_400132973
dense_33_400132975
dense_34_400133001
dense_34_400133003
normalize_3_400133036
normalize_3_400133038
dense_35_400133062
dense_35_400133064
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualbets'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallcards0model_6_400132737model_6_400132739model_6_400132741model_6_400132743*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324212!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallcards1model_7_400132772model_7_400132774model_7_400132776model_7_400132778*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326472!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimumbets+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_400132812dense_27_400132814*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_27_layer_call_and_return_conditional_losses_4001328012"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_400132838dense_30_400132840*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_30_layer_call_and_return_conditional_losses_4001328272"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_400132865dense_28_400132867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_28_layer_call_and_return_conditional_losses_4001328542"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_400132892dense_29_400132894*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_29_layer_call_and_return_conditional_losses_4001328812"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_400132918dense_31_400132920*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_31_layer_call_and_return_conditional_losses_4001329072"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_400132946dense_32_400132948*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_32_layer_call_and_return_conditional_losses_4001329352"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_400132973dense_33_400132975*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_33_layer_call_and_return_conditional_losses_4001329622"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_400133001dense_34_400133003*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_34_layer_call_and_return_conditional_losses_4001329902"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_400133036normalize_3_400133038*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_normalize_3_layer_call_and_return_conditional_losses_4001330252%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_400133062dense_35_400133064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_35_layer_call_and_return_conditional_losses_4001330512"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?-
?
F__inference_model_7_layer_call_and_return_conditional_losses_400132580
input_8+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_400132524
embedding_21_400132546
embedding_22_400132570
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinput_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_7_layer_call_and_return_conditional_losses_4001324872
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_400132524*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_23_layer_call_and_return_conditional_losses_4001325152&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_400132546*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_21_layer_call_and_return_conditional_losses_4001325372&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_400132570*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_22_layer_call_and_return_conditional_losses_4001325612&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?	
?
G__inference_dense_35_layer_call_and_return_conditional_losses_400133051

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
K__inference_embedding_22_layer_call_and_return_conditional_losses_400132561

inputs
embedding_lookup_400132555
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400132555Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400132555*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400132555*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_27_layer_call_fn_400134280

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_27_layer_call_and_return_conditional_losses_4001328012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_32_layer_call_fn_400134377

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_32_layer_call_and_return_conditional_losses_4001329352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_dense_34_layer_call_fn_400134415

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_34_layer_call_and_return_conditional_losses_4001329902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_model_6_layer_call_fn_400132477
input_7
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?
I
-__inference_flatten_7_layer_call_fn_400134533

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_7_layer_call_and_return_conditional_losses_4001324872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_30_layer_call_fn_400134319

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_30_layer_call_and_return_conditional_losses_4001328272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Y
?

M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133418

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_11_greaterequal_y
model_6_400133333
model_6_400133335
model_6_400133337
model_6_400133339
model_7_400133342
model_7_400133344
model_7_400133346
model_7_400133348/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_400133360
dense_27_400133362
dense_30_400133365
dense_30_400133367
dense_28_400133370
dense_28_400133372
dense_29_400133375
dense_29_400133377
dense_31_400133380
dense_31_400133382
dense_32_400133387
dense_32_400133389
dense_33_400133393
dense_33_400133395
dense_34_400133400
dense_34_400133402
normalize_3_400133407
normalize_3_400133409
dense_35_400133412
dense_35_400133414
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_6_400133333model_6_400133335model_6_400133337model_6_400133339*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324662!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_7_400133342model_7_400133344model_7_400133346model_7_400133348*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326922!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_400133360dense_27_400133362*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_27_layer_call_and_return_conditional_losses_4001328012"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_400133365dense_30_400133367*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_30_layer_call_and_return_conditional_losses_4001328272"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_400133370dense_28_400133372*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_28_layer_call_and_return_conditional_losses_4001328542"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_400133375dense_29_400133377*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_29_layer_call_and_return_conditional_losses_4001328812"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_400133380dense_31_400133382*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_31_layer_call_and_return_conditional_losses_4001329072"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_400133387dense_32_400133389*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_32_layer_call_and_return_conditional_losses_4001329352"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_400133393dense_33_400133395*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_33_layer_call_and_return_conditional_losses_4001329622"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_400133400dense_34_400133402*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_34_layer_call_and_return_conditional_losses_4001329902"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_400133407normalize_3_400133409*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_normalize_3_layer_call_and_return_conditional_losses_4001330252%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_400133412dense_35_400133414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_35_layer_call_and_return_conditional_losses_4001330512"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
,__inference_dense_28_layer_call_fn_400134300

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_28_layer_call_and_return_conditional_losses_4001328542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
K__inference_embedding_23_layer_call_and_return_conditional_losses_400134543

inputs
embedding_lookup_400134537
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400134537Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400134537*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400134537*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
F__inference_model_7_layer_call_and_return_conditional_losses_400132647

inputs+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_400132629
embedding_21_400132632
embedding_22_400132637
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_7_layer_call_and_return_conditional_losses_4001324872
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_400132629*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_23_layer_call_and_return_conditional_losses_4001325152&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_400132632*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_21_layer_call_and_return_conditional_losses_4001325372&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_400132637*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_22_layer_call_and_return_conditional_losses_4001325612&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
??
?#
"__inference__traced_save_400134860
file_prefix.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_embedding_20_embeddings_read_readvariableop6
2savev2_embedding_18_embeddings_read_readvariableop6
2savev2_embedding_19_embeddings_read_readvariableop6
2savev2_embedding_23_embeddings_read_readvariableop6
2savev2_embedding_21_embeddings_read_readvariableop6
2savev2_embedding_22_embeddings_read_readvariableop?
;savev2_normalize_3_normalization_3_mean_read_readvariableopC
?savev2_normalize_3_normalization_3_variance_read_readvariableop@
<savev2_normalize_3_normalization_3_count_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop=
9savev2_adam_embedding_20_embeddings_m_read_readvariableop=
9savev2_adam_embedding_18_embeddings_m_read_readvariableop=
9savev2_adam_embedding_19_embeddings_m_read_readvariableop=
9savev2_adam_embedding_23_embeddings_m_read_readvariableop=
9savev2_adam_embedding_21_embeddings_m_read_readvariableop=
9savev2_adam_embedding_22_embeddings_m_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop=
9savev2_adam_embedding_20_embeddings_v_read_readvariableop=
9savev2_adam_embedding_18_embeddings_v_read_readvariableop=
9savev2_adam_embedding_19_embeddings_v_read_readvariableop=
9savev2_adam_embedding_23_embeddings_v_read_readvariableop=
9savev2_adam_embedding_21_embeddings_v_read_readvariableop=
9savev2_adam_embedding_22_embeddings_v_read_readvariableop
savev2_const_5

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?+
value?+B?+SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_embedding_20_embeddings_read_readvariableop2savev2_embedding_18_embeddings_read_readvariableop2savev2_embedding_19_embeddings_read_readvariableop2savev2_embedding_23_embeddings_read_readvariableop2savev2_embedding_21_embeddings_read_readvariableop2savev2_embedding_22_embeddings_read_readvariableop;savev2_normalize_3_normalization_3_mean_read_readvariableop?savev2_normalize_3_normalization_3_variance_read_readvariableop<savev2_normalize_3_normalization_3_count_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop9savev2_adam_embedding_20_embeddings_m_read_readvariableop9savev2_adam_embedding_18_embeddings_m_read_readvariableop9savev2_adam_embedding_19_embeddings_m_read_readvariableop9savev2_adam_embedding_23_embeddings_m_read_readvariableop9savev2_adam_embedding_21_embeddings_m_read_readvariableop9savev2_adam_embedding_22_embeddings_m_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableop9savev2_adam_embedding_20_embeddings_v_read_readvariableop9savev2_adam_embedding_18_embeddings_v_read_readvariableop9savev2_adam_embedding_19_embeddings_v_read_readvariableop9savev2_adam_embedding_23_embeddings_v_read_readvariableop9savev2_adam_embedding_21_embeddings_v_read_readvariableop9savev2_adam_embedding_22_embeddings_v_read_readvariableopsavev2_const_5"/device:CPU:0*
_output_shapes
 *a
dtypesW
U2S		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?:: : : : : :	4?:	?:	?:	4?:	?:	?:?:?: : : :
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?:
??:?:
??:?:	?:?:
??:?:
??:?:
??:?:
??:?:
??:?:	?::	4?:	?:	?:	4?:	?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	4?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?:!(

_output_shapes	
:?:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:&-"
 
_output_shapes
:
??:!.

_output_shapes	
:?:&/"
 
_output_shapes
:
??:!0

_output_shapes	
:?:&1"
 
_output_shapes
:
??:!2

_output_shapes	
:?:%3!

_output_shapes
:	?: 4

_output_shapes
::%5!

_output_shapes
:	4?:%6!

_output_shapes
:	?:%7!

_output_shapes
:	?:%8!

_output_shapes
:	4?:%9!

_output_shapes
:	?:%:!

_output_shapes
:	?:&;"
 
_output_shapes
:
??:!<

_output_shapes	
:?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:%?!

_output_shapes
:	?:!@

_output_shapes	
:?:&A"
 
_output_shapes
:
??:!B

_output_shapes	
:?:&C"
 
_output_shapes
:
??:!D

_output_shapes	
:?:&E"
 
_output_shapes
:
??:!F

_output_shapes	
:?:&G"
 
_output_shapes
:
??:!H

_output_shapes	
:?:&I"
 
_output_shapes
:
??:!J

_output_shapes	
:?:%K!

_output_shapes
:	?: L

_output_shapes
::%M!

_output_shapes
:	4?:%N!

_output_shapes
:	?:%O!

_output_shapes
:	?:%P!

_output_shapes
:	4?:%Q!

_output_shapes
:	?:%R!

_output_shapes
:	?:S

_output_shapes
: 
?	
?
G__inference_dense_29_layer_call_and_return_conditional_losses_400134330

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_34_layer_call_and_return_conditional_losses_400132990

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?-
?
F__inference_model_6_layer_call_and_return_conditional_losses_400132354
input_7*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_400132298
embedding_18_400132320
embedding_19_400132344
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinput_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_4001322612
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_400132298*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_20_layer_call_and_return_conditional_losses_4001322892&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_400132320*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_18_layer_call_and_return_conditional_losses_4001323112&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_400132344*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_19_layer_call_and_return_conditional_losses_4001323352&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?	
?
G__inference_dense_31_layer_call_and_return_conditional_losses_400134349

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_33_layer_call_and_return_conditional_losses_400134387

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
K__inference_embedding_18_layer_call_and_return_conditional_losses_400132311

inputs
embedding_lookup_400132305
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400132305Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400132305*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400132305*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
v
0__inference_embedding_22_layer_call_fn_400134584

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_22_layer_call_and_return_conditional_losses_4001325612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
F__inference_model_6_layer_call_and_return_conditional_losses_400132386
input_7*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_400132368
embedding_18_400132371
embedding_19_400132376
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinput_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_4001322612
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_400132368*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_20_layer_call_and_return_conditional_losses_4001322892&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_400132371*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_18_layer_call_and_return_conditional_losses_4001323112&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_400132376*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_19_layer_call_and_return_conditional_losses_4001323352&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?	
?
K__inference_embedding_19_layer_call_and_return_conditional_losses_400132335

inputs
embedding_lookup_400132329
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400132329Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400132329*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400132329*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
F__inference_model_6_layer_call_and_return_conditional_losses_400132466

inputs*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_400132448
embedding_18_400132451
embedding_19_400132456
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_4001322612
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_400132448*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_20_layer_call_and_return_conditional_losses_4001322892&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_400132451*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_18_layer_call_and_return_conditional_losses_4001323112&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_400132456*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_19_layer_call_and_return_conditional_losses_4001323352&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
K__inference_embedding_19_layer_call_and_return_conditional_losses_400134515

inputs
embedding_lookup_400134509
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400134509Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400134509*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400134509*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_normalize_3_layer_call_and_return_conditional_losses_400133025
x3
/normalization_3_reshape_readvariableop_resource5
1normalization_3_reshape_1_readvariableop_resource
identity??&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape_1?
normalization_3/subSubx normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_3/truediv?
IdentityIdentitynormalization_3/truediv:z:0'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?

/__inference_normalize_3_layer_call_fn_400134441
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_normalize_3_layer_call_and_return_conditional_losses_4001330252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_namex
?	
?
K__inference_embedding_22_layer_call_and_return_conditional_losses_400134577

inputs
embedding_lookup_400134571
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400134571Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400134571*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400134571*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
F__inference_model_6_layer_call_and_return_conditional_losses_400134082

inputs*
&tf_math_greater_equal_9_greaterequal_y+
'embedding_20_embedding_lookup_400134056+
'embedding_18_embedding_lookup_400134062+
'embedding_19_embedding_lookup_400134070
identity??embedding_18/embedding_lookup?embedding_19/embedding_lookup?embedding_20/embedding_lookups
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_6/Const?
flatten_6/ReshapeReshapeinputsflatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_6/Reshape?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimumflatten_6/Reshape:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqualflatten_6/Reshape:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
embedding_20/CastCast$tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_20/Cast?
embedding_20/embedding_lookupResourceGather'embedding_20_embedding_lookup_400134056embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_20/embedding_lookup/400134056*,
_output_shapes
:??????????*
dtype02
embedding_20/embedding_lookup?
&embedding_20/embedding_lookup/IdentityIdentity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_20/embedding_lookup/400134056*,
_output_shapes
:??????????2(
&embedding_20/embedding_lookup/Identity?
(embedding_20/embedding_lookup/Identity_1Identity/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_20/embedding_lookup/Identity_1?
embedding_18/CastCast%tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_18/Cast?
embedding_18/embedding_lookupResourceGather'embedding_18_embedding_lookup_400134062embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_18/embedding_lookup/400134062*,
_output_shapes
:??????????*
dtype02
embedding_18/embedding_lookup?
&embedding_18/embedding_lookup/IdentityIdentity&embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_18/embedding_lookup/400134062*,
_output_shapes
:??????????2(
&embedding_18/embedding_lookup/Identity?
(embedding_18/embedding_lookup/Identity_1Identity/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_18/embedding_lookup/Identity_1?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV21embedding_20/embedding_lookup/Identity_1:output:01embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
embedding_19/CastCasttf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_19/Cast?
embedding_19/embedding_lookupResourceGather'embedding_19_embedding_lookup_400134070embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_19/embedding_lookup/400134070*,
_output_shapes
:??????????*
dtype02
embedding_19/embedding_lookup?
&embedding_19/embedding_lookup/IdentityIdentity&embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_19/embedding_lookup/400134070*,
_output_shapes
:??????????2(
&embedding_19/embedding_lookup/Identity?
(embedding_19/embedding_lookup/Identity_1Identity/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_19/embedding_lookup/Identity_1?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:01embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0^embedding_18/embedding_lookup^embedding_19/embedding_lookup^embedding_20/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_18/embedding_lookupembedding_18/embedding_lookup2>
embedding_19/embedding_lookupembedding_19/embedding_lookup2>
embedding_20/embedding_lookupembedding_20/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
G__inference_dense_27_layer_call_and_return_conditional_losses_400132801

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_30_layer_call_and_return_conditional_losses_400132827

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
F__inference_model_6_layer_call_and_return_conditional_losses_400132421

inputs*
&tf_math_greater_equal_9_greaterequal_y
embedding_20_400132403
embedding_18_400132406
embedding_19_400132411
identity??$embedding_18/StatefulPartitionedCall?$embedding_19/StatefulPartitionedCall?$embedding_20/StatefulPartitionedCall?
flatten_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_4001322612
flatten_6/PartitionedCall?
*tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2,
*tf.clip_by_value_9/clip_by_value/Minimum/y?
(tf.clip_by_value_9/clip_by_value/MinimumMinimum"flatten_6/PartitionedCall:output:03tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2*
(tf.clip_by_value_9/clip_by_value/Minimum?
"tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"tf.clip_by_value_9/clip_by_value/y?
 tf.clip_by_value_9/clip_by_valueMaximum,tf.clip_by_value_9/clip_by_value/Minimum:z:0+tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2"
 tf.clip_by_value_9/clip_by_value?
#tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_6/FloorDiv/y?
!tf.compat.v1.floor_div_6/FloorDivFloorDiv$tf.clip_by_value_9/clip_by_value:z:0,tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_6/FloorDiv?
$tf.math.greater_equal_9/GreaterEqualGreaterEqual"flatten_6/PartitionedCall:output:0&tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2&
$tf.math.greater_equal_9/GreaterEqual?
tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_6/FloorMod/y?
tf.math.floormod_6/FloorModFloorMod$tf.clip_by_value_9/clip_by_value:z:0&tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_6/FloorMod?
$embedding_20/StatefulPartitionedCallStatefulPartitionedCall$tf.clip_by_value_9/clip_by_value:z:0embedding_20_400132403*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_20_layer_call_and_return_conditional_losses_4001322892&
$embedding_20/StatefulPartitionedCall?
$embedding_18/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_6/FloorDiv:z:0embedding_18_400132406*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_18_layer_call_and_return_conditional_losses_4001323112&
$embedding_18/StatefulPartitionedCall?
tf.cast_9/CastCast(tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_9/Cast?
tf.__operators__.add_18/AddV2AddV2-embedding_20/StatefulPartitionedCall:output:0-embedding_18/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_18/AddV2?
$embedding_19/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_6/FloorMod:z:0embedding_19_400132411*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_19_layer_call_and_return_conditional_losses_4001323352&
$embedding_19/StatefulPartitionedCall?
tf.__operators__.add_19/AddV2AddV2!tf.__operators__.add_18/AddV2:z:0-embedding_19/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_19/AddV2?
tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_6/ExpandDims/dim?
tf.expand_dims_6/ExpandDims
ExpandDimstf.cast_9/Cast:y:0(tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_6/ExpandDims?
tf.math.multiply_6/MulMul!tf.__operators__.add_19/AddV2:z:0$tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_6/Mul?
*tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_6/Sum/reduction_indices?
tf.math.reduce_sum_6/SumSumtf.math.multiply_6/Mul:z:03tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_6/Sum?
IdentityIdentity!tf.math.reduce_sum_6/Sum:output:0%^embedding_18/StatefulPartitionedCall%^embedding_19/StatefulPartitionedCall%^embedding_20/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_18/StatefulPartitionedCall$embedding_18/StatefulPartitionedCall2L
$embedding_19/StatefulPartitionedCall$embedding_19/StatefulPartitionedCall2L
$embedding_20/StatefulPartitionedCall$embedding_20/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?-
?
F__inference_model_7_layer_call_and_return_conditional_losses_400132612
input_8+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_400132594
embedding_21_400132597
embedding_22_400132602
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinput_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_7_layer_call_and_return_conditional_losses_4001324872
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_400132594*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_23_layer_call_and_return_conditional_losses_4001325152&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_400132597*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_21_layer_call_and_return_conditional_losses_4001325372&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_400132602*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_22_layer_call_and_return_conditional_losses_4001325612&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?	
?
G__inference_dense_34_layer_call_and_return_conditional_losses_400134406

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_7_layer_call_and_return_conditional_losses_400132487

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_32_layer_call_and_return_conditional_losses_400132935

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_custom_model_3_layer_call_fn_400133322

cards0

cards1
bets
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcards0cards1betsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_3_layer_call_and_return_conditional_losses_4001332572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_model_6_layer_call_fn_400134137

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
G__inference_dense_30_layer_call_and_return_conditional_losses_400134310

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_33_layer_call_fn_400134396

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_33_layer_call_and_return_conditional_losses_4001329622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
K__inference_embedding_18_layer_call_and_return_conditional_losses_400134498

inputs
embedding_lookup_400134492
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400134492Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400134492*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400134492*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_dense_31_layer_call_and_return_conditional_losses_400132907

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_normalize_3_layer_call_and_return_conditional_losses_400134432
x3
/normalization_3_reshape_readvariableop_resource5
1normalization_3_reshape_1_readvariableop_resource
identity??&normalization_3/Reshape/ReadVariableOp?(normalization_3/Reshape_1/ReadVariableOp?
&normalization_3/Reshape/ReadVariableOpReadVariableOp/normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&normalization_3/Reshape/ReadVariableOp?
normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization_3/Reshape/shape?
normalization_3/ReshapeReshape.normalization_3/Reshape/ReadVariableOp:value:0&normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape?
(normalization_3/Reshape_1/ReadVariableOpReadVariableOp1normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(normalization_3/Reshape_1/ReadVariableOp?
normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2!
normalization_3/Reshape_1/shape?
normalization_3/Reshape_1Reshape0normalization_3/Reshape_1/ReadVariableOp:value:0(normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2
normalization_3/Reshape_1?
normalization_3/subSubx normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2
normalization_3/sub?
normalization_3/SqrtSqrt"normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2
normalization_3/Sqrt{
normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization_3/Maximum/y?
normalization_3/MaximumMaximumnormalization_3/Sqrt:y:0"normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2
normalization_3/Maximum?
normalization_3/truedivRealDivnormalization_3/sub:z:0normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2
normalization_3/truediv?
IdentityIdentitynormalization_3/truediv:z:0'^normalization_3/Reshape/ReadVariableOp)^normalization_3/Reshape_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::2P
&normalization_3/Reshape/ReadVariableOp&normalization_3/Reshape/ReadVariableOp2T
(normalization_3/Reshape_1/ReadVariableOp(normalization_3/Reshape_1/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_namex
?
v
0__inference_embedding_20_layer_call_fn_400134488

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_20_layer_call_and_return_conditional_losses_4001322892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
F__inference_model_7_layer_call_and_return_conditional_losses_400132692

inputs+
'tf_math_greater_equal_10_greaterequal_y
embedding_23_400132674
embedding_21_400132677
embedding_22_400132682
identity??$embedding_21/StatefulPartitionedCall?$embedding_22/StatefulPartitionedCall?$embedding_23/StatefulPartitionedCall?
flatten_7/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_7_layer_call_and_return_conditional_losses_4001324872
flatten_7/PartitionedCall?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimum"flatten_7/PartitionedCall:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqual"flatten_7/PartitionedCall:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
$embedding_23/StatefulPartitionedCallStatefulPartitionedCall%tf.clip_by_value_10/clip_by_value:z:0embedding_23_400132674*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_23_layer_call_and_return_conditional_losses_4001325152&
$embedding_23/StatefulPartitionedCall?
$embedding_21/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.floor_div_7/FloorDiv:z:0embedding_21_400132677*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_21_layer_call_and_return_conditional_losses_4001325372&
$embedding_21/StatefulPartitionedCall?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV2-embedding_23/StatefulPartitionedCall:output:0-embedding_21/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
$embedding_22/StatefulPartitionedCallStatefulPartitionedCalltf.math.floormod_7/FloorMod:z:0embedding_22_400132682*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_22_layer_call_and_return_conditional_losses_4001325612&
$embedding_22/StatefulPartitionedCall?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:0-embedding_22/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0%^embedding_21/StatefulPartitionedCall%^embedding_22/StatefulPartitionedCall%^embedding_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2L
$embedding_21/StatefulPartitionedCall$embedding_21/StatefulPartitionedCall2L
$embedding_22/StatefulPartitionedCall$embedding_22/StatefulPartitionedCall2L
$embedding_23/StatefulPartitionedCall$embedding_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
d
H__inference_flatten_6_layer_call_and_return_conditional_losses_400134466

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133902

inputs_0_0

inputs_0_1
inputs_1+
'tf_math_greater_equal_11_greaterequal_y2
.model_6_tf_math_greater_equal_9_greaterequal_y3
/model_6_embedding_20_embedding_lookup_4001337523
/model_6_embedding_18_embedding_lookup_4001337583
/model_6_embedding_19_embedding_lookup_4001337663
/model_7_tf_math_greater_equal_10_greaterequal_y3
/model_7_embedding_23_embedding_lookup_4001337903
/model_7_embedding_21_embedding_lookup_4001337963
/model_7_embedding_22_embedding_lookup_400133804/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource+
'dense_28_matmul_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource+
'dense_29_matmul_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource?
;normalize_3_normalization_3_reshape_readvariableop_resourceA
=normalize_3_normalization_3_reshape_1_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?dense_28/BiasAdd/ReadVariableOp?dense_28/MatMul/ReadVariableOp?dense_29/BiasAdd/ReadVariableOp?dense_29/MatMul/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?dense_30/MatMul/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?dense_32/BiasAdd/ReadVariableOp?dense_32/MatMul/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?%model_6/embedding_18/embedding_lookup?%model_6/embedding_19/embedding_lookup?%model_6/embedding_20/embedding_lookup?%model_7/embedding_21/embedding_lookup?%model_7/embedding_22/embedding_lookup?%model_7/embedding_23/embedding_lookup?2normalize_3/normalization_3/Reshape/ReadVariableOp?4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_1'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_6/flatten_6/Const?
model_6/flatten_6/ReshapeReshape
inputs_0_0 model_6/flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2
model_6/flatten_6/Reshape?
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI24
2model_6/tf.clip_by_value_9/clip_by_value/Minimum/y?
0model_6/tf.clip_by_value_9/clip_by_value/MinimumMinimum"model_6/flatten_6/Reshape:output:0;model_6/tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????22
0model_6/tf.clip_by_value_9/clip_by_value/Minimum?
*model_6/tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_6/tf.clip_by_value_9/clip_by_value/y?
(model_6/tf.clip_by_value_9/clip_by_valueMaximum4model_6/tf.clip_by_value_9/clip_by_value/Minimum:z:03model_6/tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2*
(model_6/tf.clip_by_value_9/clip_by_value?
+model_6/tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_6/tf.compat.v1.floor_div_6/FloorDiv/y?
)model_6/tf.compat.v1.floor_div_6/FloorDivFloorDiv,model_6/tf.clip_by_value_9/clip_by_value:z:04model_6/tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_6/tf.compat.v1.floor_div_6/FloorDiv?
,model_6/tf.math.greater_equal_9/GreaterEqualGreaterEqual"model_6/flatten_6/Reshape:output:0.model_6_tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2.
,model_6/tf.math.greater_equal_9/GreaterEqual?
%model_6/tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_6/tf.math.floormod_6/FloorMod/y?
#model_6/tf.math.floormod_6/FloorModFloorMod,model_6/tf.clip_by_value_9/clip_by_value:z:0.model_6/tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_6/tf.math.floormod_6/FloorMod?
model_6/embedding_20/CastCast,model_6/tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_20/Cast?
%model_6/embedding_20/embedding_lookupResourceGather/model_6_embedding_20_embedding_lookup_400133752model_6/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_6/embedding_20/embedding_lookup/400133752*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_20/embedding_lookup?
.model_6/embedding_20/embedding_lookup/IdentityIdentity.model_6/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_6/embedding_20/embedding_lookup/400133752*,
_output_shapes
:??????????20
.model_6/embedding_20/embedding_lookup/Identity?
0model_6/embedding_20/embedding_lookup/Identity_1Identity7model_6/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_20/embedding_lookup/Identity_1?
model_6/embedding_18/CastCast-model_6/tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_18/Cast?
%model_6/embedding_18/embedding_lookupResourceGather/model_6_embedding_18_embedding_lookup_400133758model_6/embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_6/embedding_18/embedding_lookup/400133758*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_18/embedding_lookup?
.model_6/embedding_18/embedding_lookup/IdentityIdentity.model_6/embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_6/embedding_18/embedding_lookup/400133758*,
_output_shapes
:??????????20
.model_6/embedding_18/embedding_lookup/Identity?
0model_6/embedding_18/embedding_lookup/Identity_1Identity7model_6/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_18/embedding_lookup/Identity_1?
model_6/tf.cast_9/CastCast0model_6/tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_6/tf.cast_9/Cast?
%model_6/tf.__operators__.add_18/AddV2AddV29model_6/embedding_20/embedding_lookup/Identity_1:output:09model_6/embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_18/AddV2?
model_6/embedding_19/CastCast'model_6/tf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_6/embedding_19/Cast?
%model_6/embedding_19/embedding_lookupResourceGather/model_6_embedding_19_embedding_lookup_400133766model_6/embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_6/embedding_19/embedding_lookup/400133766*,
_output_shapes
:??????????*
dtype02'
%model_6/embedding_19/embedding_lookup?
.model_6/embedding_19/embedding_lookup/IdentityIdentity.model_6/embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_6/embedding_19/embedding_lookup/400133766*,
_output_shapes
:??????????20
.model_6/embedding_19/embedding_lookup/Identity?
0model_6/embedding_19/embedding_lookup/Identity_1Identity7model_6/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_6/embedding_19/embedding_lookup/Identity_1?
%model_6/tf.__operators__.add_19/AddV2AddV2)model_6/tf.__operators__.add_18/AddV2:z:09model_6/embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_6/tf.__operators__.add_19/AddV2?
'model_6/tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_6/tf.expand_dims_6/ExpandDims/dim?
#model_6/tf.expand_dims_6/ExpandDims
ExpandDimsmodel_6/tf.cast_9/Cast:y:00model_6/tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_6/tf.expand_dims_6/ExpandDims?
model_6/tf.math.multiply_6/MulMul)model_6/tf.__operators__.add_19/AddV2:z:0,model_6/tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_6/tf.math.multiply_6/Mul?
2model_6/tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_6/tf.math.reduce_sum_6/Sum/reduction_indices?
 model_6/tf.math.reduce_sum_6/SumSum"model_6/tf.math.multiply_6/Mul:z:0;model_6/tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_6/tf.math.reduce_sum_6/Sum?
model_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_7/flatten_7/Const?
model_7/flatten_7/ReshapeReshape
inputs_0_1 model_7/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
model_7/flatten_7/Reshape?
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI25
3model_7/tf.clip_by_value_10/clip_by_value/Minimum/y?
1model_7/tf.clip_by_value_10/clip_by_value/MinimumMinimum"model_7/flatten_7/Reshape:output:0<model_7/tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????23
1model_7/tf.clip_by_value_10/clip_by_value/Minimum?
+model_7/tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+model_7/tf.clip_by_value_10/clip_by_value/y?
)model_7/tf.clip_by_value_10/clip_by_valueMaximum5model_7/tf.clip_by_value_10/clip_by_value/Minimum:z:04model_7/tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.clip_by_value_10/clip_by_value?
+model_7/tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2-
+model_7/tf.compat.v1.floor_div_7/FloorDiv/y?
)model_7/tf.compat.v1.floor_div_7/FloorDivFloorDiv-model_7/tf.clip_by_value_10/clip_by_value:z:04model_7/tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2+
)model_7/tf.compat.v1.floor_div_7/FloorDiv?
-model_7/tf.math.greater_equal_10/GreaterEqualGreaterEqual"model_7/flatten_7/Reshape:output:0/model_7_tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2/
-model_7/tf.math.greater_equal_10/GreaterEqual?
%model_7/tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2'
%model_7/tf.math.floormod_7/FloorMod/y?
#model_7/tf.math.floormod_7/FloorModFloorMod-model_7/tf.clip_by_value_10/clip_by_value:z:0.model_7/tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2%
#model_7/tf.math.floormod_7/FloorMod?
model_7/embedding_23/CastCast-model_7/tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_23/Cast?
%model_7/embedding_23/embedding_lookupResourceGather/model_7_embedding_23_embedding_lookup_400133790model_7/embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_7/embedding_23/embedding_lookup/400133790*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_23/embedding_lookup?
.model_7/embedding_23/embedding_lookup/IdentityIdentity.model_7/embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_7/embedding_23/embedding_lookup/400133790*,
_output_shapes
:??????????20
.model_7/embedding_23/embedding_lookup/Identity?
0model_7/embedding_23/embedding_lookup/Identity_1Identity7model_7/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_23/embedding_lookup/Identity_1?
model_7/embedding_21/CastCast-model_7/tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_21/Cast?
%model_7/embedding_21/embedding_lookupResourceGather/model_7_embedding_21_embedding_lookup_400133796model_7/embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_7/embedding_21/embedding_lookup/400133796*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_21/embedding_lookup?
.model_7/embedding_21/embedding_lookup/IdentityIdentity.model_7/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_7/embedding_21/embedding_lookup/400133796*,
_output_shapes
:??????????20
.model_7/embedding_21/embedding_lookup/Identity?
0model_7/embedding_21/embedding_lookup/Identity_1Identity7model_7/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_21/embedding_lookup/Identity_1?
model_7/tf.cast_10/CastCast1model_7/tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
model_7/tf.cast_10/Cast?
%model_7/tf.__operators__.add_20/AddV2AddV29model_7/embedding_23/embedding_lookup/Identity_1:output:09model_7/embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_20/AddV2?
model_7/embedding_22/CastCast'model_7/tf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_7/embedding_22/Cast?
%model_7/embedding_22/embedding_lookupResourceGather/model_7_embedding_22_embedding_lookup_400133804model_7/embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_7/embedding_22/embedding_lookup/400133804*,
_output_shapes
:??????????*
dtype02'
%model_7/embedding_22/embedding_lookup?
.model_7/embedding_22/embedding_lookup/IdentityIdentity.model_7/embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_7/embedding_22/embedding_lookup/400133804*,
_output_shapes
:??????????20
.model_7/embedding_22/embedding_lookup/Identity?
0model_7/embedding_22/embedding_lookup/Identity_1Identity7model_7/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????22
0model_7/embedding_22/embedding_lookup/Identity_1?
%model_7/tf.__operators__.add_21/AddV2AddV2)model_7/tf.__operators__.add_20/AddV2:z:09model_7/embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2'
%model_7/tf.__operators__.add_21/AddV2?
'model_7/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'model_7/tf.expand_dims_7/ExpandDims/dim?
#model_7/tf.expand_dims_7/ExpandDims
ExpandDimsmodel_7/tf.cast_10/Cast:y:00model_7/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2%
#model_7/tf.expand_dims_7/ExpandDims?
model_7/tf.math.multiply_7/MulMul)model_7/tf.__operators__.add_21/AddV2:z:0,model_7/tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2 
model_7/tf.math.multiply_7/Mul?
2model_7/tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2model_7/tf.math.reduce_sum_7/Sum/reduction_indices?
 model_7/tf.math.reduce_sum_7/SumSum"model_7/tf.math.multiply_7/Mul:z:0;model_7/tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2"
 model_7/tf.math.reduce_sum_7/Sum?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_1+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2)model_6/tf.math.reduce_sum_6/Sum:output:0)model_7/tf.math.reduce_sum_7/Sum:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_27/MatMul/ReadVariableOp?
dense_27/MatMulMatMultf.concat_9/concat:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/MatMul?
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_27/BiasAdd/ReadVariableOp?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_27/BiasAddt
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_27/Relu?
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_30/MatMul/ReadVariableOp?
dense_30/MatMulMatMultf.concat_10/concat:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/MatMul?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_30/BiasAdd?
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_28/MatMul/ReadVariableOp?
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/MatMul?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_28/BiasAddt
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_28/Relu?
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_29/MatMul/ReadVariableOp?
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/MatMul?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_29/BiasAddt
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_29/Relu?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMuldense_30/BiasAdd:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_31/BiasAdd
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2dense_29/Relu:activations:0dense_31/BiasAdd:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_32/MatMul/ReadVariableOp?
dense_32/MatMulMatMultf.concat_11/concat:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/MatMul?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_32/BiasAdd|
tf.nn.relu_9/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMultf.nn.relu_9/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAdd?
tf.__operators__.add_22/AddV2AddV2dense_33/BiasAdd:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMul tf.nn.relu_10/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_34/BiasAdd?
tf.__operators__.add_23/AddV2AddV2dense_34/BiasAdd:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
2normalize_3/normalization_3/Reshape/ReadVariableOpReadVariableOp;normalize_3_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype024
2normalize_3/normalization_3/Reshape/ReadVariableOp?
)normalize_3/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2+
)normalize_3/normalization_3/Reshape/shape?
#normalize_3/normalization_3/ReshapeReshape:normalize_3/normalization_3/Reshape/ReadVariableOp:value:02normalize_3/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Reshape?
4normalize_3/normalization_3/Reshape_1/ReadVariableOpReadVariableOp=normalize_3_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4normalize_3/normalization_3/Reshape_1/ReadVariableOp?
+normalize_3/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2-
+normalize_3/normalization_3/Reshape_1/shape?
%normalize_3/normalization_3/Reshape_1Reshape<normalize_3/normalization_3/Reshape_1/ReadVariableOp:value:04normalize_3/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?2'
%normalize_3/normalization_3/Reshape_1?
normalize_3/normalization_3/subSub tf.nn.relu_11/Relu:activations:0,normalize_3/normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????2!
normalize_3/normalization_3/sub?
 normalize_3/normalization_3/SqrtSqrt.normalize_3/normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?2"
 normalize_3/normalization_3/Sqrt?
%normalize_3/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%normalize_3/normalization_3/Maximum/y?
#normalize_3/normalization_3/MaximumMaximum$normalize_3/normalization_3/Sqrt:y:0.normalize_3/normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?2%
#normalize_3/normalization_3/Maximum?
#normalize_3/normalization_3/truedivRealDiv#normalize_3/normalization_3/sub:z:0'normalize_3/normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????2%
#normalize_3/normalization_3/truediv?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMul'normalize_3/normalization_3/truediv:z:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdd?
IdentityIdentitydense_35/BiasAdd:output:0 ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp&^model_6/embedding_18/embedding_lookup&^model_6/embedding_19/embedding_lookup&^model_6/embedding_20/embedding_lookup&^model_7/embedding_21/embedding_lookup&^model_7/embedding_22/embedding_lookup&^model_7/embedding_23/embedding_lookup3^normalize_3/normalization_3/Reshape/ReadVariableOp5^normalize_3/normalization_3/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp2N
%model_6/embedding_18/embedding_lookup%model_6/embedding_18/embedding_lookup2N
%model_6/embedding_19/embedding_lookup%model_6/embedding_19/embedding_lookup2N
%model_6/embedding_20/embedding_lookup%model_6/embedding_20/embedding_lookup2N
%model_7/embedding_21/embedding_lookup%model_7/embedding_21/embedding_lookup2N
%model_7/embedding_22/embedding_lookup%model_7/embedding_22/embedding_lookup2N
%model_7/embedding_23/embedding_lookup%model_7/embedding_23/embedding_lookup2h
2normalize_3/normalization_3/Reshape/ReadVariableOp2normalize_3/normalization_3/Reshape/ReadVariableOp2l
4normalize_3/normalization_3/Reshape_1/ReadVariableOp4normalize_3/normalization_3/Reshape_1/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
$__inference__wrapped_model_400132251

cards0

cards1
bets:
6custom_model_3_tf_math_greater_equal_11_greaterequal_yA
=custom_model_3_model_6_tf_math_greater_equal_9_greaterequal_yB
>custom_model_3_model_6_embedding_20_embedding_lookup_400132101B
>custom_model_3_model_6_embedding_18_embedding_lookup_400132107B
>custom_model_3_model_6_embedding_19_embedding_lookup_400132115B
>custom_model_3_model_7_tf_math_greater_equal_10_greaterequal_yB
>custom_model_3_model_7_embedding_23_embedding_lookup_400132139B
>custom_model_3_model_7_embedding_21_embedding_lookup_400132145B
>custom_model_3_model_7_embedding_22_embedding_lookup_400132153>
:custom_model_3_tf_clip_by_value_11_clip_by_value_minimum_y6
2custom_model_3_tf_clip_by_value_11_clip_by_value_y:
6custom_model_3_dense_27_matmul_readvariableop_resource;
7custom_model_3_dense_27_biasadd_readvariableop_resource:
6custom_model_3_dense_30_matmul_readvariableop_resource;
7custom_model_3_dense_30_biasadd_readvariableop_resource:
6custom_model_3_dense_28_matmul_readvariableop_resource;
7custom_model_3_dense_28_biasadd_readvariableop_resource:
6custom_model_3_dense_29_matmul_readvariableop_resource;
7custom_model_3_dense_29_biasadd_readvariableop_resource:
6custom_model_3_dense_31_matmul_readvariableop_resource;
7custom_model_3_dense_31_biasadd_readvariableop_resource:
6custom_model_3_dense_32_matmul_readvariableop_resource;
7custom_model_3_dense_32_biasadd_readvariableop_resource:
6custom_model_3_dense_33_matmul_readvariableop_resource;
7custom_model_3_dense_33_biasadd_readvariableop_resource:
6custom_model_3_dense_34_matmul_readvariableop_resource;
7custom_model_3_dense_34_biasadd_readvariableop_resourceN
Jcustom_model_3_normalize_3_normalization_3_reshape_readvariableop_resourceP
Lcustom_model_3_normalize_3_normalization_3_reshape_1_readvariableop_resource:
6custom_model_3_dense_35_matmul_readvariableop_resource;
7custom_model_3_dense_35_biasadd_readvariableop_resource
identity??.custom_model_3/dense_27/BiasAdd/ReadVariableOp?-custom_model_3/dense_27/MatMul/ReadVariableOp?.custom_model_3/dense_28/BiasAdd/ReadVariableOp?-custom_model_3/dense_28/MatMul/ReadVariableOp?.custom_model_3/dense_29/BiasAdd/ReadVariableOp?-custom_model_3/dense_29/MatMul/ReadVariableOp?.custom_model_3/dense_30/BiasAdd/ReadVariableOp?-custom_model_3/dense_30/MatMul/ReadVariableOp?.custom_model_3/dense_31/BiasAdd/ReadVariableOp?-custom_model_3/dense_31/MatMul/ReadVariableOp?.custom_model_3/dense_32/BiasAdd/ReadVariableOp?-custom_model_3/dense_32/MatMul/ReadVariableOp?.custom_model_3/dense_33/BiasAdd/ReadVariableOp?-custom_model_3/dense_33/MatMul/ReadVariableOp?.custom_model_3/dense_34/BiasAdd/ReadVariableOp?-custom_model_3/dense_34/MatMul/ReadVariableOp?.custom_model_3/dense_35/BiasAdd/ReadVariableOp?-custom_model_3/dense_35/MatMul/ReadVariableOp?4custom_model_3/model_6/embedding_18/embedding_lookup?4custom_model_3/model_6/embedding_19/embedding_lookup?4custom_model_3/model_6/embedding_20/embedding_lookup?4custom_model_3/model_7/embedding_21/embedding_lookup?4custom_model_3/model_7/embedding_22/embedding_lookup?4custom_model_3/model_7/embedding_23/embedding_lookup?Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp?Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp?
4custom_model_3/tf.math.greater_equal_11/GreaterEqualGreaterEqualbets6custom_model_3_tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
26
4custom_model_3/tf.math.greater_equal_11/GreaterEqual?
&custom_model_3/model_6/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_3/model_6/flatten_6/Const?
(custom_model_3/model_6/flatten_6/ReshapeReshapecards0/custom_model_3/model_6/flatten_6/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/flatten_6/Reshape?
Acustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2C
Acustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum/y?
?custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/MinimumMinimum1custom_model_3/model_6/flatten_6/Reshape:output:0Jcustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2A
?custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum?
9custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2;
9custom_model_3/model_6/tf.clip_by_value_9/clip_by_value/y?
7custom_model_3/model_6/tf.clip_by_value_9/clip_by_valueMaximumCcustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/Minimum:z:0Bcustom_model_3/model_6/tf.clip_by_value_9/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????29
7custom_model_3/model_6/tf.clip_by_value_9/clip_by_value?
:custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv/y?
8custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDivFloorDiv;custom_model_3/model_6/tf.clip_by_value_9/clip_by_value:z:0Ccustom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv?
;custom_model_3/model_6/tf.math.greater_equal_9/GreaterEqualGreaterEqual1custom_model_3/model_6/flatten_6/Reshape:output:0=custom_model_3_model_6_tf_math_greater_equal_9_greaterequal_y*
T0*'
_output_shapes
:?????????2=
;custom_model_3/model_6/tf.math.greater_equal_9/GreaterEqual?
4custom_model_3/model_6/tf.math.floormod_6/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_3/model_6/tf.math.floormod_6/FloorMod/y?
2custom_model_3/model_6/tf.math.floormod_6/FloorModFloorMod;custom_model_3/model_6/tf.clip_by_value_9/clip_by_value:z:0=custom_model_3/model_6/tf.math.floormod_6/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_3/model_6/tf.math.floormod_6/FloorMod?
(custom_model_3/model_6/embedding_20/CastCast;custom_model_3/model_6/tf.clip_by_value_9/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/embedding_20/Cast?
4custom_model_3/model_6/embedding_20/embedding_lookupResourceGather>custom_model_3_model_6_embedding_20_embedding_lookup_400132101,custom_model_3/model_6/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_3/model_6/embedding_20/embedding_lookup/400132101*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_6/embedding_20/embedding_lookup?
=custom_model_3/model_6/embedding_20/embedding_lookup/IdentityIdentity=custom_model_3/model_6/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_3/model_6/embedding_20/embedding_lookup/400132101*,
_output_shapes
:??????????2?
=custom_model_3/model_6/embedding_20/embedding_lookup/Identity?
?custom_model_3/model_6/embedding_20/embedding_lookup/Identity_1IdentityFcustom_model_3/model_6/embedding_20/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_6/embedding_20/embedding_lookup/Identity_1?
(custom_model_3/model_6/embedding_18/CastCast<custom_model_3/model_6/tf.compat.v1.floor_div_6/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/embedding_18/Cast?
4custom_model_3/model_6/embedding_18/embedding_lookupResourceGather>custom_model_3_model_6_embedding_18_embedding_lookup_400132107,custom_model_3/model_6/embedding_18/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_3/model_6/embedding_18/embedding_lookup/400132107*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_6/embedding_18/embedding_lookup?
=custom_model_3/model_6/embedding_18/embedding_lookup/IdentityIdentity=custom_model_3/model_6/embedding_18/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_3/model_6/embedding_18/embedding_lookup/400132107*,
_output_shapes
:??????????2?
=custom_model_3/model_6/embedding_18/embedding_lookup/Identity?
?custom_model_3/model_6/embedding_18/embedding_lookup/Identity_1IdentityFcustom_model_3/model_6/embedding_18/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_6/embedding_18/embedding_lookup/Identity_1?
%custom_model_3/model_6/tf.cast_9/CastCast?custom_model_3/model_6/tf.math.greater_equal_9/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2'
%custom_model_3/model_6/tf.cast_9/Cast?
4custom_model_3/model_6/tf.__operators__.add_18/AddV2AddV2Hcustom_model_3/model_6/embedding_20/embedding_lookup/Identity_1:output:0Hcustom_model_3/model_6/embedding_18/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_6/tf.__operators__.add_18/AddV2?
(custom_model_3/model_6/embedding_19/CastCast6custom_model_3/model_6/tf.math.floormod_6/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_6/embedding_19/Cast?
4custom_model_3/model_6/embedding_19/embedding_lookupResourceGather>custom_model_3_model_6_embedding_19_embedding_lookup_400132115,custom_model_3/model_6/embedding_19/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_3/model_6/embedding_19/embedding_lookup/400132115*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_6/embedding_19/embedding_lookup?
=custom_model_3/model_6/embedding_19/embedding_lookup/IdentityIdentity=custom_model_3/model_6/embedding_19/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_3/model_6/embedding_19/embedding_lookup/400132115*,
_output_shapes
:??????????2?
=custom_model_3/model_6/embedding_19/embedding_lookup/Identity?
?custom_model_3/model_6/embedding_19/embedding_lookup/Identity_1IdentityFcustom_model_3/model_6/embedding_19/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_6/embedding_19/embedding_lookup/Identity_1?
4custom_model_3/model_6/tf.__operators__.add_19/AddV2AddV28custom_model_3/model_6/tf.__operators__.add_18/AddV2:z:0Hcustom_model_3/model_6/embedding_19/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_6/tf.__operators__.add_19/AddV2?
6custom_model_3/model_6/tf.expand_dims_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_3/model_6/tf.expand_dims_6/ExpandDims/dim?
2custom_model_3/model_6/tf.expand_dims_6/ExpandDims
ExpandDims)custom_model_3/model_6/tf.cast_9/Cast:y:0?custom_model_3/model_6/tf.expand_dims_6/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_3/model_6/tf.expand_dims_6/ExpandDims?
-custom_model_3/model_6/tf.math.multiply_6/MulMul8custom_model_3/model_6/tf.__operators__.add_19/AddV2:z:0;custom_model_3/model_6/tf.expand_dims_6/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_3/model_6/tf.math.multiply_6/Mul?
Acustom_model_3/model_6/tf.math.reduce_sum_6/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_3/model_6/tf.math.reduce_sum_6/Sum/reduction_indices?
/custom_model_3/model_6/tf.math.reduce_sum_6/SumSum1custom_model_3/model_6/tf.math.multiply_6/Mul:z:0Jcustom_model_3/model_6/tf.math.reduce_sum_6/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_3/model_6/tf.math.reduce_sum_6/Sum?
&custom_model_3/model_7/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&custom_model_3/model_7/flatten_7/Const?
(custom_model_3/model_7/flatten_7/ReshapeReshapecards1/custom_model_3/model_7/flatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/flatten_7/Reshape?
Bcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2D
Bcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum/y?
@custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/MinimumMinimum1custom_model_3/model_7/flatten_7/Reshape:output:0Kcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2B
@custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum?
:custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:custom_model_3/model_7/tf.clip_by_value_10/clip_by_value/y?
8custom_model_3/model_7/tf.clip_by_value_10/clip_by_valueMaximumDcustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/Minimum:z:0Ccustom_model_3/model_7/tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_3/model_7/tf.clip_by_value_10/clip_by_value?
:custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2<
:custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv/y?
8custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDivFloorDiv<custom_model_3/model_7/tf.clip_by_value_10/clip_by_value:z:0Ccustom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2:
8custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv?
<custom_model_3/model_7/tf.math.greater_equal_10/GreaterEqualGreaterEqual1custom_model_3/model_7/flatten_7/Reshape:output:0>custom_model_3_model_7_tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2>
<custom_model_3/model_7/tf.math.greater_equal_10/GreaterEqual?
4custom_model_3/model_7/tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@26
4custom_model_3/model_7/tf.math.floormod_7/FloorMod/y?
2custom_model_3/model_7/tf.math.floormod_7/FloorModFloorMod<custom_model_3/model_7/tf.clip_by_value_10/clip_by_value:z:0=custom_model_3/model_7/tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????24
2custom_model_3/model_7/tf.math.floormod_7/FloorMod?
(custom_model_3/model_7/embedding_23/CastCast<custom_model_3/model_7/tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/embedding_23/Cast?
4custom_model_3/model_7/embedding_23/embedding_lookupResourceGather>custom_model_3_model_7_embedding_23_embedding_lookup_400132139,custom_model_3/model_7/embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_3/model_7/embedding_23/embedding_lookup/400132139*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_7/embedding_23/embedding_lookup?
=custom_model_3/model_7/embedding_23/embedding_lookup/IdentityIdentity=custom_model_3/model_7/embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_3/model_7/embedding_23/embedding_lookup/400132139*,
_output_shapes
:??????????2?
=custom_model_3/model_7/embedding_23/embedding_lookup/Identity?
?custom_model_3/model_7/embedding_23/embedding_lookup/Identity_1IdentityFcustom_model_3/model_7/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_7/embedding_23/embedding_lookup/Identity_1?
(custom_model_3/model_7/embedding_21/CastCast<custom_model_3/model_7/tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/embedding_21/Cast?
4custom_model_3/model_7/embedding_21/embedding_lookupResourceGather>custom_model_3_model_7_embedding_21_embedding_lookup_400132145,custom_model_3/model_7/embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_3/model_7/embedding_21/embedding_lookup/400132145*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_7/embedding_21/embedding_lookup?
=custom_model_3/model_7/embedding_21/embedding_lookup/IdentityIdentity=custom_model_3/model_7/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_3/model_7/embedding_21/embedding_lookup/400132145*,
_output_shapes
:??????????2?
=custom_model_3/model_7/embedding_21/embedding_lookup/Identity?
?custom_model_3/model_7/embedding_21/embedding_lookup/Identity_1IdentityFcustom_model_3/model_7/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_7/embedding_21/embedding_lookup/Identity_1?
&custom_model_3/model_7/tf.cast_10/CastCast@custom_model_3/model_7/tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2(
&custom_model_3/model_7/tf.cast_10/Cast?
4custom_model_3/model_7/tf.__operators__.add_20/AddV2AddV2Hcustom_model_3/model_7/embedding_23/embedding_lookup/Identity_1:output:0Hcustom_model_3/model_7/embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_7/tf.__operators__.add_20/AddV2?
(custom_model_3/model_7/embedding_22/CastCast6custom_model_3/model_7/tf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2*
(custom_model_3/model_7/embedding_22/Cast?
4custom_model_3/model_7/embedding_22/embedding_lookupResourceGather>custom_model_3_model_7_embedding_22_embedding_lookup_400132153,custom_model_3/model_7/embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Q
_classG
ECloc:@custom_model_3/model_7/embedding_22/embedding_lookup/400132153*,
_output_shapes
:??????????*
dtype026
4custom_model_3/model_7/embedding_22/embedding_lookup?
=custom_model_3/model_7/embedding_22/embedding_lookup/IdentityIdentity=custom_model_3/model_7/embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Q
_classG
ECloc:@custom_model_3/model_7/embedding_22/embedding_lookup/400132153*,
_output_shapes
:??????????2?
=custom_model_3/model_7/embedding_22/embedding_lookup/Identity?
?custom_model_3/model_7/embedding_22/embedding_lookup/Identity_1IdentityFcustom_model_3/model_7/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2A
?custom_model_3/model_7/embedding_22/embedding_lookup/Identity_1?
4custom_model_3/model_7/tf.__operators__.add_21/AddV2AddV28custom_model_3/model_7/tf.__operators__.add_20/AddV2:z:0Hcustom_model_3/model_7/embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????26
4custom_model_3/model_7/tf.__operators__.add_21/AddV2?
6custom_model_3/model_7/tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????28
6custom_model_3/model_7/tf.expand_dims_7/ExpandDims/dim?
2custom_model_3/model_7/tf.expand_dims_7/ExpandDims
ExpandDims*custom_model_3/model_7/tf.cast_10/Cast:y:0?custom_model_3/model_7/tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????24
2custom_model_3/model_7/tf.expand_dims_7/ExpandDims?
-custom_model_3/model_7/tf.math.multiply_7/MulMul8custom_model_3/model_7/tf.__operators__.add_21/AddV2:z:0;custom_model_3/model_7/tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2/
-custom_model_3/model_7/tf.math.multiply_7/Mul?
Acustom_model_3/model_7/tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2C
Acustom_model_3/model_7/tf.math.reduce_sum_7/Sum/reduction_indices?
/custom_model_3/model_7/tf.math.reduce_sum_7/SumSum1custom_model_3/model_7/tf.math.multiply_7/Mul:z:0Jcustom_model_3/model_7/tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????21
/custom_model_3/model_7/tf.math.reduce_sum_7/Sum?
8custom_model_3/tf.clip_by_value_11/clip_by_value/MinimumMinimumbets:custom_model_3_tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2:
8custom_model_3/tf.clip_by_value_11/clip_by_value/Minimum?
0custom_model_3/tf.clip_by_value_11/clip_by_valueMaximum<custom_model_3/tf.clip_by_value_11/clip_by_value/Minimum:z:02custom_model_3_tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
22
0custom_model_3/tf.clip_by_value_11/clip_by_value?
custom_model_3/tf.cast_11/CastCast8custom_model_3/tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2 
custom_model_3/tf.cast_11/Cast?
&custom_model_3/tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&custom_model_3/tf.concat_9/concat/axis?
!custom_model_3/tf.concat_9/concatConcatV28custom_model_3/model_6/tf.math.reduce_sum_6/Sum:output:08custom_model_3/model_7/tf.math.reduce_sum_7/Sum:output:0/custom_model_3/tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2#
!custom_model_3/tf.concat_9/concat?
'custom_model_3/tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_3/tf.concat_10/concat/axis?
"custom_model_3/tf.concat_10/concatConcatV24custom_model_3/tf.clip_by_value_11/clip_by_value:z:0"custom_model_3/tf.cast_11/Cast:y:00custom_model_3/tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2$
"custom_model_3/tf.concat_10/concat?
-custom_model_3/dense_27/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_27_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_27/MatMul/ReadVariableOp?
custom_model_3/dense_27/MatMulMatMul*custom_model_3/tf.concat_9/concat:output:05custom_model_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_27/MatMul?
.custom_model_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_27/BiasAdd/ReadVariableOp?
custom_model_3/dense_27/BiasAddBiasAdd(custom_model_3/dense_27/MatMul:product:06custom_model_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_27/BiasAdd?
custom_model_3/dense_27/ReluRelu(custom_model_3/dense_27/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_3/dense_27/Relu?
-custom_model_3/dense_30/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_30_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_3/dense_30/MatMul/ReadVariableOp?
custom_model_3/dense_30/MatMulMatMul+custom_model_3/tf.concat_10/concat:output:05custom_model_3/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_30/MatMul?
.custom_model_3/dense_30/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_30/BiasAdd/ReadVariableOp?
custom_model_3/dense_30/BiasAddBiasAdd(custom_model_3/dense_30/MatMul:product:06custom_model_3/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_30/BiasAdd?
-custom_model_3/dense_28/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_28_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_28/MatMul/ReadVariableOp?
custom_model_3/dense_28/MatMulMatMul*custom_model_3/dense_27/Relu:activations:05custom_model_3/dense_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_28/MatMul?
.custom_model_3/dense_28/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_28/BiasAdd/ReadVariableOp?
custom_model_3/dense_28/BiasAddBiasAdd(custom_model_3/dense_28/MatMul:product:06custom_model_3/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_28/BiasAdd?
custom_model_3/dense_28/ReluRelu(custom_model_3/dense_28/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_3/dense_28/Relu?
-custom_model_3/dense_29/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_29_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_29/MatMul/ReadVariableOp?
custom_model_3/dense_29/MatMulMatMul*custom_model_3/dense_28/Relu:activations:05custom_model_3/dense_29/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_29/MatMul?
.custom_model_3/dense_29/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_29/BiasAdd/ReadVariableOp?
custom_model_3/dense_29/BiasAddBiasAdd(custom_model_3/dense_29/MatMul:product:06custom_model_3/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_29/BiasAdd?
custom_model_3/dense_29/ReluRelu(custom_model_3/dense_29/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
custom_model_3/dense_29/Relu?
-custom_model_3/dense_31/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_31/MatMul/ReadVariableOp?
custom_model_3/dense_31/MatMulMatMul(custom_model_3/dense_30/BiasAdd:output:05custom_model_3/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_31/MatMul?
.custom_model_3/dense_31/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_31/BiasAdd/ReadVariableOp?
custom_model_3/dense_31/BiasAddBiasAdd(custom_model_3/dense_31/MatMul:product:06custom_model_3/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_31/BiasAdd?
'custom_model_3/tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'custom_model_3/tf.concat_11/concat/axis?
"custom_model_3/tf.concat_11/concatConcatV2*custom_model_3/dense_29/Relu:activations:0(custom_model_3/dense_31/BiasAdd:output:00custom_model_3/tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2$
"custom_model_3/tf.concat_11/concat?
-custom_model_3/dense_32/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_32/MatMul/ReadVariableOp?
custom_model_3/dense_32/MatMulMatMul+custom_model_3/tf.concat_11/concat:output:05custom_model_3/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_32/MatMul?
.custom_model_3/dense_32/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_32/BiasAdd/ReadVariableOp?
custom_model_3/dense_32/BiasAddBiasAdd(custom_model_3/dense_32/MatMul:product:06custom_model_3/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_32/BiasAdd?
 custom_model_3/tf.nn.relu_9/ReluRelu(custom_model_3/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2"
 custom_model_3/tf.nn.relu_9/Relu?
-custom_model_3/dense_33/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_33/MatMul/ReadVariableOp?
custom_model_3/dense_33/MatMulMatMul.custom_model_3/tf.nn.relu_9/Relu:activations:05custom_model_3/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_33/MatMul?
.custom_model_3/dense_33/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_33/BiasAdd/ReadVariableOp?
custom_model_3/dense_33/BiasAddBiasAdd(custom_model_3/dense_33/MatMul:product:06custom_model_3/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_33/BiasAdd?
,custom_model_3/tf.__operators__.add_22/AddV2AddV2(custom_model_3/dense_33/BiasAdd:output:0.custom_model_3/tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_3/tf.__operators__.add_22/AddV2?
!custom_model_3/tf.nn.relu_10/ReluRelu0custom_model_3/tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_3/tf.nn.relu_10/Relu?
-custom_model_3/dense_34/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_34_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02/
-custom_model_3/dense_34/MatMul/ReadVariableOp?
custom_model_3/dense_34/MatMulMatMul/custom_model_3/tf.nn.relu_10/Relu:activations:05custom_model_3/dense_34/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
custom_model_3/dense_34/MatMul?
.custom_model_3/dense_34/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_34_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.custom_model_3/dense_34/BiasAdd/ReadVariableOp?
custom_model_3/dense_34/BiasAddBiasAdd(custom_model_3/dense_34/MatMul:product:06custom_model_3/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
custom_model_3/dense_34/BiasAdd?
,custom_model_3/tf.__operators__.add_23/AddV2AddV2(custom_model_3/dense_34/BiasAdd:output:0/custom_model_3/tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2.
,custom_model_3/tf.__operators__.add_23/AddV2?
!custom_model_3/tf.nn.relu_11/ReluRelu0custom_model_3/tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2#
!custom_model_3/tf.nn.relu_11/Relu?
Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOpReadVariableOpJcustom_model_3_normalize_3_normalization_3_reshape_readvariableop_resource*
_output_shapes	
:?*
dtype02C
Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp?
8custom_model_3/normalize_3/normalization_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8custom_model_3/normalize_3/normalization_3/Reshape/shape?
2custom_model_3/normalize_3/normalization_3/ReshapeReshapeIcustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp:value:0Acustom_model_3/normalize_3/normalization_3/Reshape/shape:output:0*
T0*
_output_shapes
:	?24
2custom_model_3/normalize_3/normalization_3/Reshape?
Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOpReadVariableOpLcustom_model_3_normalize_3_normalization_3_reshape_1_readvariableop_resource*
_output_shapes	
:?*
dtype02E
Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp?
:custom_model_3/normalize_3/normalization_3/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:custom_model_3/normalize_3/normalization_3/Reshape_1/shape?
4custom_model_3/normalize_3/normalization_3/Reshape_1ReshapeKcustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp:value:0Ccustom_model_3/normalize_3/normalization_3/Reshape_1/shape:output:0*
T0*
_output_shapes
:	?26
4custom_model_3/normalize_3/normalization_3/Reshape_1?
.custom_model_3/normalize_3/normalization_3/subSub/custom_model_3/tf.nn.relu_11/Relu:activations:0;custom_model_3/normalize_3/normalization_3/Reshape:output:0*
T0*(
_output_shapes
:??????????20
.custom_model_3/normalize_3/normalization_3/sub?
/custom_model_3/normalize_3/normalization_3/SqrtSqrt=custom_model_3/normalize_3/normalization_3/Reshape_1:output:0*
T0*
_output_shapes
:	?21
/custom_model_3/normalize_3/normalization_3/Sqrt?
4custom_model_3/normalize_3/normalization_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???326
4custom_model_3/normalize_3/normalization_3/Maximum/y?
2custom_model_3/normalize_3/normalization_3/MaximumMaximum3custom_model_3/normalize_3/normalization_3/Sqrt:y:0=custom_model_3/normalize_3/normalization_3/Maximum/y:output:0*
T0*
_output_shapes
:	?24
2custom_model_3/normalize_3/normalization_3/Maximum?
2custom_model_3/normalize_3/normalization_3/truedivRealDiv2custom_model_3/normalize_3/normalization_3/sub:z:06custom_model_3/normalize_3/normalization_3/Maximum:z:0*
T0*(
_output_shapes
:??????????24
2custom_model_3/normalize_3/normalization_3/truediv?
-custom_model_3/dense_35/MatMul/ReadVariableOpReadVariableOp6custom_model_3_dense_35_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-custom_model_3/dense_35/MatMul/ReadVariableOp?
custom_model_3/dense_35/MatMulMatMul6custom_model_3/normalize_3/normalization_3/truediv:z:05custom_model_3/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
custom_model_3/dense_35/MatMul?
.custom_model_3/dense_35/BiasAdd/ReadVariableOpReadVariableOp7custom_model_3_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.custom_model_3/dense_35/BiasAdd/ReadVariableOp?
custom_model_3/dense_35/BiasAddBiasAdd(custom_model_3/dense_35/MatMul:product:06custom_model_3/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
custom_model_3/dense_35/BiasAdd?
IdentityIdentity(custom_model_3/dense_35/BiasAdd:output:0/^custom_model_3/dense_27/BiasAdd/ReadVariableOp.^custom_model_3/dense_27/MatMul/ReadVariableOp/^custom_model_3/dense_28/BiasAdd/ReadVariableOp.^custom_model_3/dense_28/MatMul/ReadVariableOp/^custom_model_3/dense_29/BiasAdd/ReadVariableOp.^custom_model_3/dense_29/MatMul/ReadVariableOp/^custom_model_3/dense_30/BiasAdd/ReadVariableOp.^custom_model_3/dense_30/MatMul/ReadVariableOp/^custom_model_3/dense_31/BiasAdd/ReadVariableOp.^custom_model_3/dense_31/MatMul/ReadVariableOp/^custom_model_3/dense_32/BiasAdd/ReadVariableOp.^custom_model_3/dense_32/MatMul/ReadVariableOp/^custom_model_3/dense_33/BiasAdd/ReadVariableOp.^custom_model_3/dense_33/MatMul/ReadVariableOp/^custom_model_3/dense_34/BiasAdd/ReadVariableOp.^custom_model_3/dense_34/MatMul/ReadVariableOp/^custom_model_3/dense_35/BiasAdd/ReadVariableOp.^custom_model_3/dense_35/MatMul/ReadVariableOp5^custom_model_3/model_6/embedding_18/embedding_lookup5^custom_model_3/model_6/embedding_19/embedding_lookup5^custom_model_3/model_6/embedding_20/embedding_lookup5^custom_model_3/model_7/embedding_21/embedding_lookup5^custom_model_3/model_7/embedding_22/embedding_lookup5^custom_model_3/model_7/embedding_23/embedding_lookupB^custom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOpD^custom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2`
.custom_model_3/dense_27/BiasAdd/ReadVariableOp.custom_model_3/dense_27/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_27/MatMul/ReadVariableOp-custom_model_3/dense_27/MatMul/ReadVariableOp2`
.custom_model_3/dense_28/BiasAdd/ReadVariableOp.custom_model_3/dense_28/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_28/MatMul/ReadVariableOp-custom_model_3/dense_28/MatMul/ReadVariableOp2`
.custom_model_3/dense_29/BiasAdd/ReadVariableOp.custom_model_3/dense_29/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_29/MatMul/ReadVariableOp-custom_model_3/dense_29/MatMul/ReadVariableOp2`
.custom_model_3/dense_30/BiasAdd/ReadVariableOp.custom_model_3/dense_30/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_30/MatMul/ReadVariableOp-custom_model_3/dense_30/MatMul/ReadVariableOp2`
.custom_model_3/dense_31/BiasAdd/ReadVariableOp.custom_model_3/dense_31/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_31/MatMul/ReadVariableOp-custom_model_3/dense_31/MatMul/ReadVariableOp2`
.custom_model_3/dense_32/BiasAdd/ReadVariableOp.custom_model_3/dense_32/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_32/MatMul/ReadVariableOp-custom_model_3/dense_32/MatMul/ReadVariableOp2`
.custom_model_3/dense_33/BiasAdd/ReadVariableOp.custom_model_3/dense_33/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_33/MatMul/ReadVariableOp-custom_model_3/dense_33/MatMul/ReadVariableOp2`
.custom_model_3/dense_34/BiasAdd/ReadVariableOp.custom_model_3/dense_34/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_34/MatMul/ReadVariableOp-custom_model_3/dense_34/MatMul/ReadVariableOp2`
.custom_model_3/dense_35/BiasAdd/ReadVariableOp.custom_model_3/dense_35/BiasAdd/ReadVariableOp2^
-custom_model_3/dense_35/MatMul/ReadVariableOp-custom_model_3/dense_35/MatMul/ReadVariableOp2l
4custom_model_3/model_6/embedding_18/embedding_lookup4custom_model_3/model_6/embedding_18/embedding_lookup2l
4custom_model_3/model_6/embedding_19/embedding_lookup4custom_model_3/model_6/embedding_19/embedding_lookup2l
4custom_model_3/model_6/embedding_20/embedding_lookup4custom_model_3/model_6/embedding_20/embedding_lookup2l
4custom_model_3/model_7/embedding_21/embedding_lookup4custom_model_3/model_7/embedding_21/embedding_lookup2l
4custom_model_3/model_7/embedding_22/embedding_lookup4custom_model_3/model_7/embedding_22/embedding_lookup2l
4custom_model_3/model_7/embedding_23/embedding_lookup4custom_model_3/model_7/embedding_23/embedding_lookup2?
Acustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOpAcustom_model_3/normalize_3/normalization_3/Reshape/ReadVariableOp2?
Ccustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOpCcustom_model_3/normalize_3/normalization_3/Reshape_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_namecards0:OK
'
_output_shapes
:?????????
 
_user_specified_namecards1:MI
'
_output_shapes
:?????????


_user_specified_namebets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
2__inference_custom_model_3_layer_call_fn_400134040

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_3_layer_call_and_return_conditional_losses_4001334182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
v
0__inference_embedding_19_layer_call_fn_400134522

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_19_layer_call_and_return_conditional_losses_4001323352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_flatten_6_layer_call_fn_400134471

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *Q
fLRJ
H__inference_flatten_6_layer_call_and_return_conditional_losses_4001322612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_model_7_layer_call_fn_400132658
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:

_output_shapes
: 
?
?
2__inference_custom_model_3_layer_call_fn_400133971

inputs_0_0

inputs_0_1
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_0_0
inputs_0_1inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !*2
config_proto" 

CPU

GPU2 *0J 8? *V
fQRO
M__inference_custom_model_3_layer_call_and_return_conditional_losses_4001332572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/0:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/0/1:QM
'
_output_shapes
:?????????

"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
v
0__inference_embedding_21_layer_call_fn_400134567

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_21_layer_call_and_return_conditional_losses_4001325372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
F__inference_model_7_layer_call_and_return_conditional_losses_400134234

inputs+
'tf_math_greater_equal_10_greaterequal_y+
'embedding_23_embedding_lookup_400134208+
'embedding_21_embedding_lookup_400134214+
'embedding_22_embedding_lookup_400134222
identity??embedding_21/embedding_lookup?embedding_22/embedding_lookup?embedding_23/embedding_lookups
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_7/Const?
flatten_7/ReshapeReshapeinputsflatten_7/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_7/Reshape?
+tf.clip_by_value_10/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * $tI2-
+tf.clip_by_value_10/clip_by_value/Minimum/y?
)tf.clip_by_value_10/clip_by_value/MinimumMinimumflatten_7/Reshape:output:04tf.clip_by_value_10/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2+
)tf.clip_by_value_10/clip_by_value/Minimum?
#tf.clip_by_value_10/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#tf.clip_by_value_10/clip_by_value/y?
!tf.clip_by_value_10/clip_by_valueMaximum-tf.clip_by_value_10/clip_by_value/Minimum:z:0,tf.clip_by_value_10/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.clip_by_value_10/clip_by_value?
#tf.compat.v1.floor_div_7/FloorDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2%
#tf.compat.v1.floor_div_7/FloorDiv/y?
!tf.compat.v1.floor_div_7/FloorDivFloorDiv%tf.clip_by_value_10/clip_by_value:z:0,tf.compat.v1.floor_div_7/FloorDiv/y:output:0*
T0*'
_output_shapes
:?????????2#
!tf.compat.v1.floor_div_7/FloorDiv?
%tf.math.greater_equal_10/GreaterEqualGreaterEqualflatten_7/Reshape:output:0'tf_math_greater_equal_10_greaterequal_y*
T0*'
_output_shapes
:?????????2'
%tf.math.greater_equal_10/GreaterEqual?
tf.math.floormod_7/FloorMod/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.floormod_7/FloorMod/y?
tf.math.floormod_7/FloorModFloorMod%tf.clip_by_value_10/clip_by_value:z:0&tf.math.floormod_7/FloorMod/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.floormod_7/FloorMod?
embedding_23/CastCast%tf.clip_by_value_10/clip_by_value:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_23/Cast?
embedding_23/embedding_lookupResourceGather'embedding_23_embedding_lookup_400134208embedding_23/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_23/embedding_lookup/400134208*,
_output_shapes
:??????????*
dtype02
embedding_23/embedding_lookup?
&embedding_23/embedding_lookup/IdentityIdentity&embedding_23/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_23/embedding_lookup/400134208*,
_output_shapes
:??????????2(
&embedding_23/embedding_lookup/Identity?
(embedding_23/embedding_lookup/Identity_1Identity/embedding_23/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_23/embedding_lookup/Identity_1?
embedding_21/CastCast%tf.compat.v1.floor_div_7/FloorDiv:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_21/Cast?
embedding_21/embedding_lookupResourceGather'embedding_21_embedding_lookup_400134214embedding_21/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_21/embedding_lookup/400134214*,
_output_shapes
:??????????*
dtype02
embedding_21/embedding_lookup?
&embedding_21/embedding_lookup/IdentityIdentity&embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_21/embedding_lookup/400134214*,
_output_shapes
:??????????2(
&embedding_21/embedding_lookup/Identity?
(embedding_21/embedding_lookup/Identity_1Identity/embedding_21/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_21/embedding_lookup/Identity_1?
tf.cast_10/CastCast)tf.math.greater_equal_10/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
tf.cast_10/Cast?
tf.__operators__.add_20/AddV2AddV21embedding_23/embedding_lookup/Identity_1:output:01embedding_21/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_20/AddV2?
embedding_22/CastCasttf.math.floormod_7/FloorMod:z:0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_22/Cast?
embedding_22/embedding_lookupResourceGather'embedding_22_embedding_lookup_400134222embedding_22/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@embedding_22/embedding_lookup/400134222*,
_output_shapes
:??????????*
dtype02
embedding_22/embedding_lookup?
&embedding_22/embedding_lookup/IdentityIdentity&embedding_22/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@embedding_22/embedding_lookup/400134222*,
_output_shapes
:??????????2(
&embedding_22/embedding_lookup/Identity?
(embedding_22/embedding_lookup/Identity_1Identity/embedding_22/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2*
(embedding_22/embedding_lookup/Identity_1?
tf.__operators__.add_21/AddV2AddV2!tf.__operators__.add_20/AddV2:z:01embedding_22/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:??????????2
tf.__operators__.add_21/AddV2?
tf.expand_dims_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
tf.expand_dims_7/ExpandDims/dim?
tf.expand_dims_7/ExpandDims
ExpandDimstf.cast_10/Cast:y:0(tf.expand_dims_7/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2
tf.expand_dims_7/ExpandDims?
tf.math.multiply_7/MulMul!tf.__operators__.add_21/AddV2:z:0$tf.expand_dims_7/ExpandDims:output:0*
T0*,
_output_shapes
:??????????2
tf.math.multiply_7/Mul?
*tf.math.reduce_sum_7/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*tf.math.reduce_sum_7/Sum/reduction_indices?
tf.math.reduce_sum_7/SumSumtf.math.multiply_7/Mul:z:03tf.math.reduce_sum_7/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
tf.math.reduce_sum_7/Sum?
IdentityIdentity!tf.math.reduce_sum_7/Sum:output:0^embedding_21/embedding_lookup^embedding_22/embedding_lookup^embedding_23/embedding_lookup*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::2>
embedding_21/embedding_lookupembedding_21/embedding_lookup2>
embedding_22/embedding_lookupembedding_22/embedding_lookup2>
embedding_23/embedding_lookupembedding_23/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_model_7_layer_call_fn_400134260

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
v
0__inference_embedding_18_layer_call_fn_400134505

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *T
fORM
K__inference_embedding_18_layer_call_and_return_conditional_losses_4001323112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_model_6_layer_call_fn_400132432
input_7
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:?????????: :::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:

_output_shapes
: 
?	
?
K__inference_embedding_23_layer_call_and_return_conditional_losses_400132515

inputs
embedding_lookup_400132509
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_400132509Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*-
_class#
!loc:@embedding_lookup/400132509*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*-
_class#
!loc:@embedding_lookup/400132509*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Y
?

M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133257

inputs
inputs_1
inputs_2+
'tf_math_greater_equal_11_greaterequal_y
model_6_400133172
model_6_400133174
model_6_400133176
model_6_400133178
model_7_400133181
model_7_400133183
model_7_400133185
model_7_400133187/
+tf_clip_by_value_11_clip_by_value_minimum_y'
#tf_clip_by_value_11_clip_by_value_y
dense_27_400133199
dense_27_400133201
dense_30_400133204
dense_30_400133206
dense_28_400133209
dense_28_400133211
dense_29_400133214
dense_29_400133216
dense_31_400133219
dense_31_400133221
dense_32_400133226
dense_32_400133228
dense_33_400133232
dense_33_400133234
dense_34_400133239
dense_34_400133241
normalize_3_400133246
normalize_3_400133248
dense_35_400133251
dense_35_400133253
identity?? dense_27/StatefulPartitionedCall? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?model_6/StatefulPartitionedCall?model_7/StatefulPartitionedCall?#normalize_3/StatefulPartitionedCall?
%tf.math.greater_equal_11/GreaterEqualGreaterEqualinputs_2'tf_math_greater_equal_11_greaterequal_y*
T0*'
_output_shapes
:?????????
2'
%tf.math.greater_equal_11/GreaterEqual?
model_6/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_6_400133172model_6_400133174model_6_400133176model_6_400133178*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_6_layer_call_and_return_conditional_losses_4001324212!
model_6/StatefulPartitionedCall?
model_7/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_7_400133181model_7_400133183model_7_400133185model_7_400133187*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_model_7_layer_call_and_return_conditional_losses_4001326472!
model_7/StatefulPartitionedCall?
)tf.clip_by_value_11/clip_by_value/MinimumMinimuminputs_2+tf_clip_by_value_11_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????
2+
)tf.clip_by_value_11/clip_by_value/Minimum?
!tf.clip_by_value_11/clip_by_valueMaximum-tf.clip_by_value_11/clip_by_value/Minimum:z:0#tf_clip_by_value_11_clip_by_value_y*
T0*'
_output_shapes
:?????????
2#
!tf.clip_by_value_11/clip_by_value?
tf.cast_11/CastCast)tf.math.greater_equal_11/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????
2
tf.cast_11/Castt
tf.concat_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
tf.concat_9/concat/axis?
tf.concat_9/concatConcatV2(model_6/StatefulPartitionedCall:output:0(model_7/StatefulPartitionedCall:output:0 tf.concat_9/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_9/concat
tf.concat_10/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_10/concat/axis?
tf.concat_10/concatConcatV2%tf.clip_by_value_11/clip_by_value:z:0tf.cast_11/Cast:y:0!tf.concat_10/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
tf.concat_10/concat?
 dense_27/StatefulPartitionedCallStatefulPartitionedCalltf.concat_9/concat:output:0dense_27_400133199dense_27_400133201*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_27_layer_call_and_return_conditional_losses_4001328012"
 dense_27/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCalltf.concat_10/concat:output:0dense_30_400133204dense_30_400133206*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_30_layer_call_and_return_conditional_losses_4001328272"
 dense_30/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_400133209dense_28_400133211*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_28_layer_call_and_return_conditional_losses_4001328542"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_400133214dense_29_400133216*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_29_layer_call_and_return_conditional_losses_4001328812"
 dense_29/StatefulPartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_400133219dense_31_400133221*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_31_layer_call_and_return_conditional_losses_4001329072"
 dense_31/StatefulPartitionedCall
tf.concat_11/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
tf.concat_11/concat/axis?
tf.concat_11/concatConcatV2)dense_29/StatefulPartitionedCall:output:0)dense_31/StatefulPartitionedCall:output:0!tf.concat_11/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
tf.concat_11/concat?
 dense_32/StatefulPartitionedCallStatefulPartitionedCalltf.concat_11/concat:output:0dense_32_400133226dense_32_400133228*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_32_layer_call_and_return_conditional_losses_4001329352"
 dense_32/StatefulPartitionedCall?
tf.nn.relu_9/ReluRelu)dense_32/StatefulPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_9/Relu?
 dense_33/StatefulPartitionedCallStatefulPartitionedCalltf.nn.relu_9/Relu:activations:0dense_33_400133232dense_33_400133234*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_33_layer_call_and_return_conditional_losses_4001329622"
 dense_33/StatefulPartitionedCall?
tf.__operators__.add_22/AddV2AddV2)dense_33/StatefulPartitionedCall:output:0tf.nn.relu_9/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_22/AddV2?
tf.nn.relu_10/ReluRelu!tf.__operators__.add_22/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_10/Relu?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_10/Relu:activations:0dense_34_400133239dense_34_400133241*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_34_layer_call_and_return_conditional_losses_4001329902"
 dense_34/StatefulPartitionedCall?
tf.__operators__.add_23/AddV2AddV2)dense_34/StatefulPartitionedCall:output:0 tf.nn.relu_10/Relu:activations:0*
T0*(
_output_shapes
:??????????2
tf.__operators__.add_23/AddV2?
tf.nn.relu_11/ReluRelu!tf.__operators__.add_23/AddV2:z:0*
T0*(
_output_shapes
:??????????2
tf.nn.relu_11/Relu?
#normalize_3/StatefulPartitionedCallStatefulPartitionedCall tf.nn.relu_11/Relu:activations:0normalize_3_400133246normalize_3_400133248*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_normalize_3_layer_call_and_return_conditional_losses_4001330252%
#normalize_3/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall,normalize_3/StatefulPartitionedCall:output:0dense_35_400133251dense_35_400133253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *P
fKRI
G__inference_dense_35_layer_call_and_return_conditional_losses_4001330512"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall ^model_6/StatefulPartitionedCall ^model_7/StatefulPartitionedCall$^normalize_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????
: : :::: :::: : ::::::::::::::::::::2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2B
model_6/StatefulPartitionedCallmodel_6/StatefulPartitionedCall2B
model_7/StatefulPartitionedCallmodel_7/StatefulPartitionedCall2J
#normalize_3/StatefulPartitionedCall#normalize_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?,
%__inference__traced_restore_400135116
file_prefix$
 assignvariableop_dense_27_kernel$
 assignvariableop_1_dense_27_bias&
"assignvariableop_2_dense_28_kernel$
 assignvariableop_3_dense_28_bias&
"assignvariableop_4_dense_30_kernel$
 assignvariableop_5_dense_30_bias&
"assignvariableop_6_dense_29_kernel$
 assignvariableop_7_dense_29_bias&
"assignvariableop_8_dense_31_kernel$
 assignvariableop_9_dense_31_bias'
#assignvariableop_10_dense_32_kernel%
!assignvariableop_11_dense_32_bias'
#assignvariableop_12_dense_33_kernel%
!assignvariableop_13_dense_33_bias'
#assignvariableop_14_dense_34_kernel%
!assignvariableop_15_dense_34_bias'
#assignvariableop_16_dense_35_kernel%
!assignvariableop_17_dense_35_bias!
assignvariableop_18_adam_iter#
assignvariableop_19_adam_beta_1#
assignvariableop_20_adam_beta_2"
assignvariableop_21_adam_decay*
&assignvariableop_22_adam_learning_rate/
+assignvariableop_23_embedding_20_embeddings/
+assignvariableop_24_embedding_18_embeddings/
+assignvariableop_25_embedding_19_embeddings/
+assignvariableop_26_embedding_23_embeddings/
+assignvariableop_27_embedding_21_embeddings/
+assignvariableop_28_embedding_22_embeddings8
4assignvariableop_29_normalize_3_normalization_3_mean<
8assignvariableop_30_normalize_3_normalization_3_variance9
5assignvariableop_31_normalize_3_normalization_3_count
assignvariableop_32_total
assignvariableop_33_count.
*assignvariableop_34_adam_dense_27_kernel_m,
(assignvariableop_35_adam_dense_27_bias_m.
*assignvariableop_36_adam_dense_28_kernel_m,
(assignvariableop_37_adam_dense_28_bias_m.
*assignvariableop_38_adam_dense_30_kernel_m,
(assignvariableop_39_adam_dense_30_bias_m.
*assignvariableop_40_adam_dense_29_kernel_m,
(assignvariableop_41_adam_dense_29_bias_m.
*assignvariableop_42_adam_dense_31_kernel_m,
(assignvariableop_43_adam_dense_31_bias_m.
*assignvariableop_44_adam_dense_32_kernel_m,
(assignvariableop_45_adam_dense_32_bias_m.
*assignvariableop_46_adam_dense_33_kernel_m,
(assignvariableop_47_adam_dense_33_bias_m.
*assignvariableop_48_adam_dense_34_kernel_m,
(assignvariableop_49_adam_dense_34_bias_m.
*assignvariableop_50_adam_dense_35_kernel_m,
(assignvariableop_51_adam_dense_35_bias_m6
2assignvariableop_52_adam_embedding_20_embeddings_m6
2assignvariableop_53_adam_embedding_18_embeddings_m6
2assignvariableop_54_adam_embedding_19_embeddings_m6
2assignvariableop_55_adam_embedding_23_embeddings_m6
2assignvariableop_56_adam_embedding_21_embeddings_m6
2assignvariableop_57_adam_embedding_22_embeddings_m.
*assignvariableop_58_adam_dense_27_kernel_v,
(assignvariableop_59_adam_dense_27_bias_v.
*assignvariableop_60_adam_dense_28_kernel_v,
(assignvariableop_61_adam_dense_28_bias_v.
*assignvariableop_62_adam_dense_30_kernel_v,
(assignvariableop_63_adam_dense_30_bias_v.
*assignvariableop_64_adam_dense_29_kernel_v,
(assignvariableop_65_adam_dense_29_bias_v.
*assignvariableop_66_adam_dense_31_kernel_v,
(assignvariableop_67_adam_dense_31_bias_v.
*assignvariableop_68_adam_dense_32_kernel_v,
(assignvariableop_69_adam_dense_32_bias_v.
*assignvariableop_70_adam_dense_33_kernel_v,
(assignvariableop_71_adam_dense_33_bias_v.
*assignvariableop_72_adam_dense_34_kernel_v,
(assignvariableop_73_adam_dense_34_bias_v.
*assignvariableop_74_adam_dense_35_kernel_v,
(assignvariableop_75_adam_dense_35_bias_v6
2assignvariableop_76_adam_embedding_20_embeddings_v6
2assignvariableop_77_adam_embedding_18_embeddings_v6
2assignvariableop_78_adam_embedding_19_embeddings_v6
2assignvariableop_79_adam_embedding_23_embeddings_v6
2assignvariableop_80_adam_embedding_21_embeddings_v6
2assignvariableop_81_adam_embedding_22_embeddings_v
identity_83??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_9?,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?+
value?+B?+SB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:S*
dtype0*?
value?B?SB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*a
dtypesW
U2S		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_27_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_30_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_30_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_29_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_29_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_31_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_31_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_32_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_32_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_33_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_33_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_34_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_34_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_35_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_35_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_embedding_20_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_embedding_18_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_embedding_19_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_embedding_23_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_embedding_21_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_embedding_22_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_normalize_3_normalization_3_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp8assignvariableop_30_normalize_3_normalization_3_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_normalize_3_normalization_3_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_dense_27_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_dense_27_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_dense_28_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_dense_28_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_30_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_30_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_29_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_29_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_dense_31_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_dense_31_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_dense_32_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_dense_32_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_dense_33_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_dense_33_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_dense_34_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_dense_34_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_dense_35_kernel_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_dense_35_bias_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_embedding_20_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_embedding_18_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_embedding_19_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_embedding_23_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_embedding_21_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_embedding_22_embeddings_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_dense_27_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp(assignvariableop_59_adam_dense_27_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_dense_28_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_dense_28_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_dense_30_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp(assignvariableop_63_adam_dense_30_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_dense_29_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_dense_29_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_dense_31_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adam_dense_31_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp*assignvariableop_68_adam_dense_32_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_dense_32_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_dense_33_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dense_33_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_dense_34_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dense_34_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_dense_35_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_dense_35_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_embedding_20_embeddings_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp2assignvariableop_77_adam_embedding_18_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp2assignvariableop_78_adam_embedding_19_embeddings_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp2assignvariableop_79_adam_embedding_23_embeddings_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp2assignvariableop_80_adam_embedding_21_embeddings_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_embedding_22_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_819
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_82Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_82?
Identity_83IdentityIdentity_82:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_83"#
identity_83Identity_83:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
5
bets-
serving_default_bets:0?????????

9
cards0/
serving_default_cards0:0?????????
9
cards1/
serving_default_cards1:0?????????<
dense_350
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
Ϣ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer_with_weights-9
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
	optimizer
loss
	variables
regularization_losses
trainable_variables
 	keras_api
!
signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"ڛ
_tf_keras_network??{"class_name": "CustomModel", "name": "custom_model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "custom_model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}, "name": "model_6", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}, "name": "model_7", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_11", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_11", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_9", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_9", "inbound_nodes": [[["model_6", 1, 0, {"axis": 1}], ["model_7", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_11", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_11", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_11", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_11", "inbound_nodes": [["tf.math.greater_equal_11", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["tf.concat_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_10", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_10", "inbound_nodes": [[["tf.clip_by_value_11", 0, 0, {"axis": -1}], ["tf.cast_11", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["tf.concat_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dense_30", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_11", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_11", "inbound_nodes": [[["dense_29", 0, 0, {"axis": -1}], ["dense_31", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["tf.concat_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_9", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_9", "inbound_nodes": [["dense_32", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["tf.nn.relu_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_22", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_22", "inbound_nodes": [["dense_33", 0, 0, {"y": ["tf.nn.relu_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_10", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_10", "inbound_nodes": [["tf.__operators__.add_22", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["tf.nn.relu_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_23", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_23", "inbound_nodes": [["dense_34", 0, 0, {"y": ["tf.nn.relu_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_11", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_11", "inbound_nodes": [["tf.__operators__.add_23", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_3", "trainable": true, "dtype": "float32"}, "name": "normalize_3", "inbound_nodes": [[["tf.nn.relu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["normalize_3", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_35", 0, 0]]}, "build_input_shape": [[{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 3]}], {"class_name": "TensorShape", "items": [null, 10]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CustomModel", "config": {"name": "custom_model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}, "name": "cards0", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}, "name": "cards1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}, "name": "bets", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}, "name": "model_6", "inbound_nodes": [[["cards0", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}, "name": "model_7", "inbound_nodes": [[["cards1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_11", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_11", "inbound_nodes": [["bets", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_9", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_9", "inbound_nodes": [[["model_6", 1, 0, {"axis": 1}], ["model_7", 1, 0, {"axis": 1}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_11", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_11", "inbound_nodes": [["bets", 0, 0, {"clip_value_min": 0.0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_11", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_11", "inbound_nodes": [["tf.math.greater_equal_11", 0, 0, {"dtype": "float32"}]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["tf.concat_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_10", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_10", "inbound_nodes": [[["tf.clip_by_value_11", 0, 0, {"axis": -1}], ["tf.cast_11", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_28", "inbound_nodes": [[["dense_27", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["tf.concat_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_29", "inbound_nodes": [[["dense_28", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dense_30", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.concat_11", "trainable": true, "dtype": "float32", "function": "concat"}, "name": "tf.concat_11", "inbound_nodes": [[["dense_29", 0, 0, {"axis": -1}], ["dense_31", 0, 0, {"axis": -1}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["tf.concat_11", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_9", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_9", "inbound_nodes": [["dense_32", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["tf.nn.relu_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_22", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_22", "inbound_nodes": [["dense_33", 0, 0, {"y": ["tf.nn.relu_9", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_10", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_10", "inbound_nodes": [["tf.__operators__.add_22", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["tf.nn.relu_10", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_23", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_23", "inbound_nodes": [["dense_34", 0, 0, {"y": ["tf.nn.relu_10", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.nn.relu_11", "trainable": true, "dtype": "float32", "function": "nn.relu"}, "name": "tf.nn.relu_11", "inbound_nodes": [["tf.__operators__.add_23", 0, 0, {"name": null}]]}, {"class_name": "Normalize", "config": {"name": "normalize_3", "trainable": true, "dtype": "float32"}, "name": "normalize_3", "inbound_nodes": [[["tf.nn.relu_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["normalize_3", 0, 0, {}]]]}], "input_layers": [[["cards0", 0, 0], ["cards1", 0, 0]], ["bets", 0, 0]], "output_layers": [["dense_35", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0020000000949949026, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards0", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards0"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "cards1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "cards1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "bets", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "bets"}}
?Q
"layer-0
#layer-1
$layer-2
%layer-3
&layer_with_weights-0
&layer-4
'layer_with_weights-1
'layer-5
(layer-6
)layer-7
*layer-8
+layer_with_weights-2
+layer-9
,layer-10
-layer-11
.layer-12
/layer-13
0layer-14
1	variables
2regularization_losses
3trainable_variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_9", "inbound_nodes": [["flatten_6", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_20", "inbound_nodes": [[["tf.clip_by_value_9", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_18", "inbound_nodes": [[["tf.compat.v1.floor_div_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_6", "inbound_nodes": [["tf.clip_by_value_9", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_9", "inbound_nodes": [["flatten_6", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_18", "inbound_nodes": [["embedding_20", 0, 0, {"y": ["embedding_18", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_19", "inbound_nodes": [[["tf.math.floormod_6", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_9", "inbound_nodes": [["tf.math.greater_equal_9", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_19", "inbound_nodes": [["tf.__operators__.add_18", 0, 0, {"y": ["embedding_19", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_6", "inbound_nodes": [["tf.cast_9", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["tf.__operators__.add_19", 0, 0, {"y": ["tf.expand_dims_6", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_6", "inbound_nodes": [["tf.math.multiply_6", 0, 0, {"axis": 1}]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["tf.math.reduce_sum_6", 0, 0]]}}}
?Q
5layer-0
6layer-1
7layer-2
8layer-3
9layer_with_weights-0
9layer-4
:layer_with_weights-1
:layer-5
;layer-6
<layer-7
=layer-8
>layer_with_weights-2
>layer-9
?layer-10
@layer-11
Alayer-12
Blayer-13
Clayer-14
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?N
_tf_keras_network?N{"class_name": "Functional", "name": "model_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value_10", "inbound_nodes": [["flatten_7", 0, 0, {"clip_value_min": 0, "clip_value_max": 1000000.0}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}, "name": "tf.compat.v1.floor_div_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_23", "inbound_nodes": [[["tf.clip_by_value_10", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_21", "inbound_nodes": [[["tf.compat.v1.floor_div_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}, "name": "tf.math.floormod_7", "inbound_nodes": [["tf.clip_by_value_10", 0, 0, {"y": 4, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}, "name": "tf.math.greater_equal_10", "inbound_nodes": [["flatten_7", 0, 0, {"y": 0.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_20", "inbound_nodes": [["embedding_23", 0, 0, {"y": ["embedding_21", 0, 0], "name": null}]]}, {"class_name": "Embedding", "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "embedding_22", "inbound_nodes": [[["tf.math.floormod_7", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}, "name": "tf.cast_10", "inbound_nodes": [["tf.math.greater_equal_10", 0, 0, {"dtype": "float32"}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_21", "inbound_nodes": [["tf.__operators__.add_20", 0, 0, {"y": ["embedding_22", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}, "name": "tf.expand_dims_7", "inbound_nodes": [["tf.cast_10", 0, 0, {"axis": -1}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["tf.__operators__.add_21", 0, 0, {"y": ["tf.expand_dims_7", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum_7", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {"axis": 1}]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["tf.math.reduce_sum_7", 0, 0]]}}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_11", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_9", "trainable": true, "dtype": "float32", "function": "concat"}}
?
J	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_11", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
K	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_11", "trainable": true, "dtype": "float32", "function": "cast"}}
?

Lkernel
Mbias
N	variables
Oregularization_losses
Ptrainable_variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
R	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_10", "trainable": true, "dtype": "float32", "function": "concat"}}
?

Skernel
Tbias
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

Ykernel
Zbias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

_kernel
`bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

ekernel
fbias
g	variables
hregularization_losses
itrainable_variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
k	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.concat_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.concat_11", "trainable": true, "dtype": "float32", "function": "concat"}}
?

lkernel
mbias
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
r	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_9", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

skernel
tbias
u	variables
vregularization_losses
wtrainable_variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
y	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_22", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
z	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_10", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?

{kernel
|bias
}	variables
~regularization_losses
trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_23", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.nn.relu_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.nn.relu_11", "trainable": true, "dtype": "float32", "function": "nn.relu"}}
?
?	normalize
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Normalize", "name": "normalize_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_3", "trainable": true, "dtype": "float32"}}
?
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": 0}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateLm?Mm?Sm?Tm?Ym?Zm?_m?`m?em?fm?lm?mm?sm?tm?{m?|m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Lv?Mv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?lv?mv?sv?tv?{v?|v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23
?24
?25
?26"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
L6
M7
S8
T9
Y10
Z11
_12
`13
e14
f15
l16
m17
s18
t19
{20
|21
?22
?23"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
	variables
?metrics
regularization_losses
?layer_metrics
?non_trainable_variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_9", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_6", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_6", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_9", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_18", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_9", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_19", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_6", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_6", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
1	variables
?metrics
2regularization_losses
?layer_metrics
?non_trainable_variables
3trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_7", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value_10", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.floor_div_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.floor_div_7", "trainable": true, "dtype": "float32", "function": "compat.v1.floor_div"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 52, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 13, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.floormod_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.floormod_7", "trainable": true, "dtype": "float32", "function": "math.floormod"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.greater_equal_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.greater_equal_10", "trainable": true, "dtype": "float32", "function": "math.greater_equal"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_20", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?
embeddings
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 256, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.cast_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.cast_10", "trainable": true, "dtype": "float32", "function": "cast"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_21", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.expand_dims_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.expand_dims_7", "trainable": true, "dtype": "float32", "function": "expand_dims"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum_7", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
D	variables
?metrics
Eregularization_losses
?layer_metrics
?non_trainable_variables
Ftrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
??2dense_27/kernel
:?2dense_27/bias
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
N	variables
?metrics
Oregularization_losses
?layer_metrics
?non_trainable_variables
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_28/kernel
:?2dense_28/bias
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
U	variables
?metrics
Vregularization_losses
?layer_metrics
?non_trainable_variables
Wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_30/kernel
:?2dense_30/bias
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
[	variables
?metrics
\regularization_losses
?layer_metrics
?non_trainable_variables
]trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_29/kernel
:?2dense_29/bias
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
a	variables
?metrics
bregularization_losses
?layer_metrics
?non_trainable_variables
ctrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_31/kernel
:?2dense_31/bias
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
g	variables
?metrics
hregularization_losses
?layer_metrics
?non_trainable_variables
itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_32/kernel
:?2dense_32/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
n	variables
?metrics
oregularization_losses
?layer_metrics
?non_trainable_variables
ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
#:!
??2dense_33/kernel
:?2dense_33/bias
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
u	variables
?metrics
vregularization_losses
?layer_metrics
?non_trainable_variables
wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
#:!
??2dense_34/kernel
:?2dense_34/bias
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
}	variables
?metrics
~regularization_losses
?layer_metrics
?non_trainable_variables
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
?
?state_variables
?_broadcast_shape
	?mean
?variance

?count
?	keras_api"?
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization_3", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_35/kernel
:2dense_35/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(	4?2embedding_20/embeddings
*:(	?2embedding_18/embeddings
*:(	?2embedding_19/embeddings
*:(	4?2embedding_23/embeddings
*:(	?2embedding_21/embeddings
*:(	?2embedding_22/embeddings
-:+?2 normalize_3/normalization_3/mean
1:/?2$normalize_3/normalization_3/variance
):'	 2!normalize_3/normalization_3/count
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?	variables
?metrics
?regularization_losses
?layer_metrics
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
?
50
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
F
	?mean
?variance

?count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
(:&
??2Adam/dense_27/kernel/m
!:?2Adam/dense_27/bias/m
(:&
??2Adam/dense_28/kernel/m
!:?2Adam/dense_28/bias/m
':%	?2Adam/dense_30/kernel/m
!:?2Adam/dense_30/bias/m
(:&
??2Adam/dense_29/kernel/m
!:?2Adam/dense_29/bias/m
(:&
??2Adam/dense_31/kernel/m
!:?2Adam/dense_31/bias/m
(:&
??2Adam/dense_32/kernel/m
!:?2Adam/dense_32/bias/m
(:&
??2Adam/dense_33/kernel/m
!:?2Adam/dense_33/bias/m
(:&
??2Adam/dense_34/kernel/m
!:?2Adam/dense_34/bias/m
':%	?2Adam/dense_35/kernel/m
 :2Adam/dense_35/bias/m
/:-	4?2Adam/embedding_20/embeddings/m
/:-	?2Adam/embedding_18/embeddings/m
/:-	?2Adam/embedding_19/embeddings/m
/:-	4?2Adam/embedding_23/embeddings/m
/:-	?2Adam/embedding_21/embeddings/m
/:-	?2Adam/embedding_22/embeddings/m
(:&
??2Adam/dense_27/kernel/v
!:?2Adam/dense_27/bias/v
(:&
??2Adam/dense_28/kernel/v
!:?2Adam/dense_28/bias/v
':%	?2Adam/dense_30/kernel/v
!:?2Adam/dense_30/bias/v
(:&
??2Adam/dense_29/kernel/v
!:?2Adam/dense_29/bias/v
(:&
??2Adam/dense_31/kernel/v
!:?2Adam/dense_31/bias/v
(:&
??2Adam/dense_32/kernel/v
!:?2Adam/dense_32/bias/v
(:&
??2Adam/dense_33/kernel/v
!:?2Adam/dense_33/bias/v
(:&
??2Adam/dense_34/kernel/v
!:?2Adam/dense_34/bias/v
':%	?2Adam/dense_35/kernel/v
 :2Adam/dense_35/bias/v
/:-	4?2Adam/embedding_20/embeddings/v
/:-	?2Adam/embedding_18/embeddings/v
/:-	?2Adam/embedding_19/embeddings/v
/:-	4?2Adam/embedding_23/embeddings/v
/:-	?2Adam/embedding_21/embeddings/v
/:-	?2Adam/embedding_22/embeddings/v
?2?
$__inference__wrapped_model_400132251?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

?2?
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133902
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133732
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133068
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133160?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_custom_model_3_layer_call_fn_400134040
2__inference_custom_model_3_layer_call_fn_400133483
2__inference_custom_model_3_layer_call_fn_400133322
2__inference_custom_model_3_layer_call_fn_400133971?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_6_layer_call_and_return_conditional_losses_400132386
F__inference_model_6_layer_call_and_return_conditional_losses_400134082
F__inference_model_6_layer_call_and_return_conditional_losses_400134124
F__inference_model_6_layer_call_and_return_conditional_losses_400132354?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_model_6_layer_call_fn_400132477
+__inference_model_6_layer_call_fn_400134150
+__inference_model_6_layer_call_fn_400134137
+__inference_model_6_layer_call_fn_400132432?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_7_layer_call_and_return_conditional_losses_400132612
F__inference_model_7_layer_call_and_return_conditional_losses_400134234
F__inference_model_7_layer_call_and_return_conditional_losses_400134192
F__inference_model_7_layer_call_and_return_conditional_losses_400132580?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_model_7_layer_call_fn_400134260
+__inference_model_7_layer_call_fn_400132703
+__inference_model_7_layer_call_fn_400134247
+__inference_model_7_layer_call_fn_400132658?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dense_27_layer_call_and_return_conditional_losses_400134271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_27_layer_call_fn_400134280?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_28_layer_call_and_return_conditional_losses_400134291?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_28_layer_call_fn_400134300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_30_layer_call_and_return_conditional_losses_400134310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_30_layer_call_fn_400134319?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_29_layer_call_and_return_conditional_losses_400134330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_29_layer_call_fn_400134339?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_31_layer_call_and_return_conditional_losses_400134349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_31_layer_call_fn_400134358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_32_layer_call_and_return_conditional_losses_400134368?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_32_layer_call_fn_400134377?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_33_layer_call_and_return_conditional_losses_400134387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_33_layer_call_fn_400134396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_34_layer_call_and_return_conditional_losses_400134406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_34_layer_call_fn_400134415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_normalize_3_layer_call_and_return_conditional_losses_400134432?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_normalize_3_layer_call_fn_400134441?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_35_layer_call_and_return_conditional_losses_400134451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_35_layer_call_fn_400134460?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_400133562betscards0cards1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_6_layer_call_and_return_conditional_losses_400134466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_6_layer_call_fn_400134471?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_embedding_20_layer_call_and_return_conditional_losses_400134481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_embedding_20_layer_call_fn_400134488?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_embedding_18_layer_call_and_return_conditional_losses_400134498?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_embedding_18_layer_call_fn_400134505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_embedding_19_layer_call_and_return_conditional_losses_400134515?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_embedding_19_layer_call_fn_400134522?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_7_layer_call_and_return_conditional_losses_400134528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_7_layer_call_fn_400134533?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_embedding_23_layer_call_and_return_conditional_losses_400134543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_embedding_23_layer_call_fn_400134550?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_embedding_21_layer_call_and_return_conditional_losses_400134560?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_embedding_21_layer_call_fn_400134567?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_embedding_22_layer_call_and_return_conditional_losses_400134577?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_embedding_22_layer_call_fn_400134584?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4?
$__inference__wrapped_model_400132251?.???????????LMYZST_`eflmst{|????{?x
q?n
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

? "3?0
.
dense_35"?
dense_35??????????
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133068?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "%?"
?
0?????????
? ?
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133160?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "%?"
?
0?????????
? ?
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133732?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "%?"
?
0?????????
? ?
M__inference_custom_model_3_layer_call_and_return_conditional_losses_400133902?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "%?"
?
0?????????
? ?
2__inference_custom_model_3_layer_call_fn_400133322?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p

 
? "???????????
2__inference_custom_model_3_layer_call_fn_400133483?.???????????LMYZST_`eflmst{|???????
y?v
l?i
G?D
 ?
cards0?????????
 ?
cards1?????????
?
bets?????????

p 

 
? "???????????
2__inference_custom_model_3_layer_call_fn_400133971?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p

 
? "???????????
2__inference_custom_model_3_layer_call_fn_400134040?.???????????LMYZST_`eflmst{|???????
???
x?u
O?L
$?!

inputs/0/0?????????
$?!

inputs/0/1?????????
"?
inputs/1?????????

p 

 
? "???????????
G__inference_dense_27_layer_call_and_return_conditional_losses_400134271^LM0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_27_layer_call_fn_400134280QLM0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_28_layer_call_and_return_conditional_losses_400134291^ST0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_28_layer_call_fn_400134300QST0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_29_layer_call_and_return_conditional_losses_400134330^_`0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_29_layer_call_fn_400134339Q_`0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_30_layer_call_and_return_conditional_losses_400134310]YZ/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_dense_30_layer_call_fn_400134319PYZ/?,
%?"
 ?
inputs?????????
? "????????????
G__inference_dense_31_layer_call_and_return_conditional_losses_400134349^ef0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_31_layer_call_fn_400134358Qef0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_32_layer_call_and_return_conditional_losses_400134368^lm0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_32_layer_call_fn_400134377Qlm0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_33_layer_call_and_return_conditional_losses_400134387^st0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_33_layer_call_fn_400134396Qst0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_34_layer_call_and_return_conditional_losses_400134406^{|0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_dense_34_layer_call_fn_400134415Q{|0?-
&?#
!?
inputs??????????
? "????????????
G__inference_dense_35_layer_call_and_return_conditional_losses_400134451_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
,__inference_dense_35_layer_call_fn_400134460R??0?-
&?#
!?
inputs??????????
? "???????????
K__inference_embedding_18_layer_call_and_return_conditional_losses_400134498a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_18_layer_call_fn_400134505T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_19_layer_call_and_return_conditional_losses_400134515a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_19_layer_call_fn_400134522T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_20_layer_call_and_return_conditional_losses_400134481a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_20_layer_call_fn_400134488T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_21_layer_call_and_return_conditional_losses_400134560a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_21_layer_call_fn_400134567T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_22_layer_call_and_return_conditional_losses_400134577a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_22_layer_call_fn_400134584T?/?,
%?"
 ?
inputs?????????
? "????????????
K__inference_embedding_23_layer_call_and_return_conditional_losses_400134543a?/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_embedding_23_layer_call_fn_400134550T?/?,
%?"
 ?
inputs?????????
? "????????????
H__inference_flatten_6_layer_call_and_return_conditional_losses_400134466X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_flatten_6_layer_call_fn_400134471K/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_flatten_7_layer_call_and_return_conditional_losses_400134528X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_flatten_7_layer_call_fn_400134533K/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_model_6_layer_call_and_return_conditional_losses_400132354l????8?5
.?+
!?
input_7?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_6_layer_call_and_return_conditional_losses_400132386l????8?5
.?+
!?
input_7?????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_6_layer_call_and_return_conditional_losses_400134082k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_6_layer_call_and_return_conditional_losses_400134124k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
+__inference_model_6_layer_call_fn_400132432_????8?5
.?+
!?
input_7?????????
p

 
? "????????????
+__inference_model_6_layer_call_fn_400132477_????8?5
.?+
!?
input_7?????????
p 

 
? "????????????
+__inference_model_6_layer_call_fn_400134137^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
+__inference_model_6_layer_call_fn_400134150^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
F__inference_model_7_layer_call_and_return_conditional_losses_400132580l????8?5
.?+
!?
input_8?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_7_layer_call_and_return_conditional_losses_400132612l????8?5
.?+
!?
input_8?????????
p 

 
? "&?#
?
0??????????
? ?
F__inference_model_7_layer_call_and_return_conditional_losses_400134192k????7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????
? ?
F__inference_model_7_layer_call_and_return_conditional_losses_400134234k????7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
+__inference_model_7_layer_call_fn_400132658_????8?5
.?+
!?
input_8?????????
p

 
? "????????????
+__inference_model_7_layer_call_fn_400132703_????8?5
.?+
!?
input_8?????????
p 

 
? "????????????
+__inference_model_7_layer_call_fn_400134247^????7?4
-?*
 ?
inputs?????????
p

 
? "????????????
+__inference_model_7_layer_call_fn_400134260^????7?4
-?*
 ?
inputs?????????
p 

 
? "????????????
J__inference_normalize_3_layer_call_and_return_conditional_losses_400134432[??+?(
!?
?
x??????????
? "&?#
?
0??????????
? ?
/__inference_normalize_3_layer_call_fn_400134441N??+?(
!?
?
x??????????
? "????????????
'__inference_signature_wrapper_400133562?.???????????LMYZST_`eflmst{|???????
? 
???
&
bets?
bets?????????

*
cards0 ?
cards0?????????
*
cards1 ?
cards1?????????"3?0
.
dense_35"?
dense_35?????????