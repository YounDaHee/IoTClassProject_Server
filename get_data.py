import socket
import learn_data as model
import os

HOST = "Server Addr"  # 로컬호스트
PORT = "Server Port"      # 포트포워딩으로 매핑한 PC 측 포트

# train에 들어있는 txt 파일을 기반으로 학습 진행
# 서버를 실행하고 대략 2초 정도만 실행되고, 이후엔 클래스에 저장된 모델 이용
SVM = model.LearnData()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"Server wating... {HOST}:{PORT}")

try : 
    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connected by {client_address}")

        try: 
            while True:
                option = client_socket.recv(1024)
                if not option :
                    break
                option = option.decode().split("#")
                print(option)
                
                # 클라이언트가 50초 간 모은 데이터를 전달. 서버는 이를 txt 파일로 저장하여 사용.
                # 수집한 정보는 train 폴더에 저장
                if(option[0]=="0") :
                    file_name = option[1]
                    file_path = os.path.join("train", file_name)

                    with open(file_path, "wb") as file:
                        while True:
                            data = client_socket.recv(1024)
                            if not data:
                                break
                            file.write(data)
                    
                    # 새로 들어온 데이터를 기반으로 학습 재 진행
                    SVM = model.LearnData()

                # 클라이언트가 현재 와이파이 정보 보내주면, 서버가 모델 돌려서 결과 반환
                elif(option[0]=="1") : 
                    length = option[1]
                    if not length:
                        print("client disconnected")
                        break

                    # 와이파이 정보 수신
                    length =  int(length)
                    data = b""
                    while len(data) < length:
                        chunk = client_socket.recv(1024)
                        if not chunk:
                            break
                        data += chunk
                    response = data.decode()
                    
                    # 모델 수행(SVM 객체)
                    position = SVM.detect_position(response)
                    print(position)

                    # 클라이언트에게 결과 반환
                    client_socket.send((position).encode())
        except Exception as e:
            print(f"Error during client handling: {e}")
        finally:
            # 클라이언트 소켓 닫기
            client_socket.close()
            print(f"Connection with {client_address} closed.")
except KeyboardInterrupt:
    print("\nServer shutting down...")
finally:
    # 서버 소켓 닫기
    server_socket.close()
    print("Server socket closed.")

